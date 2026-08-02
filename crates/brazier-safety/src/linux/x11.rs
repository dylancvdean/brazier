use std::{
    io::{Read as _, Write as _},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result, bail};
use x11rb::{
    COPY_DEPTH_FROM_PARENT,
    connection::Connection,
    protocol::{
        Event,
        randr::ConnectionExt as _,
        shape::{ConnectionExt as _, SK, SO},
        xinput::{ConnectionExt as _, Device, EventMask as XIEventSelection, XIEventMask},
        xproto::{
            AtomEnum, ChangeGCAux, ConfigureWindowAux, ConnectionExt as _, CreateGCAux,
            CreateWindowAux, EventMask, PropMode, Rectangle, StackMode, WindowClass,
        },
    },
    rust_connection::RustConnection,
    wrapper::ConnectionExt as _,
};

const WIDTH: u16 = 420;
const HEIGHT: u16 = 54;
const ESCAPE_KEYSYM: u32 = 0xff1b;

pub fn run() -> Result<()> {
    let alive = Arc::new(AtomicBool::new(true));
    let alive_on_pipe = Arc::clone(&alive);
    thread::Builder::new()
        .name("brazier-safety-parent-watch".to_owned())
        .spawn(move || {
            let mut byte = [0_u8; 1];
            let mut input = std::io::stdin();
            loop {
                match input.read(&mut byte) {
                    Ok(0) | Err(_) => {
                        alive_on_pipe.store(false, Ordering::Release);
                        return;
                    }
                    Ok(_) => {}
                }
            }
        })
        .context("spawn safety parent watcher")?;

    let (connection, screen_index) = x11rb::connect(None).context("connect to X11")?;
    let screen = &connection.setup().roots[screen_index];
    let root = screen.root;
    let root_width = screen.width_in_pixels;
    let escape_keycode = escape_keycode(&connection)?;

    connection
        .xinput_xi_query_version(2, 0)?
        .reply()
        .context("XInput2 is required for the Esc emergency stop")?;
    connection
        .xinput_xi_select_events(
            root,
            &[XIEventSelection {
                deviceid: Device::ALL_MASTER.into(),
                mask: vec![XIEventMask::RAW_KEY_PRESS],
            }],
        )?
        .check()
        .context("watch raw X11 keyboard input")?;

    let window = connection.generate_id()?;
    let (x, y) = overlay_position(&connection, root, root_width);
    connection
        .create_window(
            COPY_DEPTH_FROM_PARENT,
            window,
            root,
            x,
            y,
            WIDTH,
            HEIGHT,
            0,
            WindowClass::INPUT_OUTPUT,
            0,
            &CreateWindowAux::new()
                .override_redirect(1)
                .background_pixel(0x31140e)
                .event_mask(EventMask::EXPOSURE | EventMask::VISIBILITY_CHANGE),
        )?
        .check()
        .context("create X11 safety overlay")?;

    set_window_properties(&connection, window)?;
    // An empty input shape makes the overlay click-through while XInput2 raw
    // key observation remains attached to the root window.
    connection
        .shape_rectangles(
            SO::SET,
            SK::INPUT,
            x11rb::protocol::xproto::ClipOrdering::UNSORTED,
            window,
            0,
            0,
            &[],
        )?
        .check()
        .context("make X11 safety overlay click-through")?;

    let font = connection.generate_id()?;
    connection
        .open_font(font, b"9x15bold")?
        .check()
        .or_else(|_| connection.open_font(font, b"fixed")?.check())
        .context("open an X11 safety overlay font")?;
    let gc = connection.generate_id()?;
    connection
        .create_gc(
            gc,
            window,
            &CreateGCAux::new()
                .foreground(0xffffff)
                .background(0x31140e)
                .font(font),
        )?
        .check()?;
    connection.map_window(window)?.check()?;
    restack(&connection, window, x, y)?;
    draw(&connection, window, gc)?;
    connection.flush()?;
    // A reply is a round trip: READY is not emitted until the server has
    // processed mapping, drawing, input shape, and raw-key selection.
    connection.get_input_focus()?.reply()?;
    println!("READY");
    std::io::stdout().flush()?;

    let mut last_restack = Instant::now();
    while alive.load(Ordering::Acquire) {
        while let Some(event) = connection.poll_for_event()? {
            match event {
                Event::XinputRawKeyPress(event) if event.detail == u32::from(escape_keycode) => {
                    println!("ESC");
                    std::io::stdout().flush()?;
                    return Ok(());
                }
                Event::Expose(_) | Event::VisibilityNotify(_) => {
                    draw(&connection, window, gc)?;
                    let (x, y) = overlay_position(&connection, root, root_width);
                    restack(&connection, window, x, y)?;
                }
                _ => {}
            }
        }
        if last_restack.elapsed() >= Duration::from_millis(100) {
            let (x, y) = overlay_position(&connection, root, root_width);
            restack(&connection, window, x, y)?;
            connection.flush()?;
            last_restack = Instant::now();
        }
        thread::sleep(Duration::from_millis(10));
    }
    bail!("safety parent closed its control pipe")
}

fn escape_keycode(connection: &RustConnection) -> Result<u8> {
    let setup = connection.setup();
    let first = setup.min_keycode;
    let count = setup.max_keycode - first + 1;
    let mapping = connection.get_keyboard_mapping(first, count)?.reply()?;
    let width = mapping.keysyms_per_keycode as usize;
    mapping
        .keysyms
        .chunks(width)
        .position(|symbols| symbols.contains(&ESCAPE_KEYSYM))
        .and_then(|offset| u8::try_from(usize::from(first) + offset).ok())
        .context("the X11 keyboard map has no Escape key")
}

fn set_window_properties(connection: &RustConnection, window: u32) -> Result<()> {
    let window_type = connection
        .intern_atom(false, b"_NET_WM_WINDOW_TYPE")?
        .reply()?
        .atom;
    let notification = connection
        .intern_atom(false, b"_NET_WM_WINDOW_TYPE_NOTIFICATION")?
        .reply()?
        .atom;
    connection.change_property32(
        PropMode::REPLACE,
        window,
        window_type,
        AtomEnum::ATOM,
        &[notification],
    )?;
    let state = connection
        .intern_atom(false, b"_NET_WM_STATE")?
        .reply()?
        .atom;
    let above = connection
        .intern_atom(false, b"_NET_WM_STATE_ABOVE")?
        .reply()?
        .atom;
    let sticky = connection
        .intern_atom(false, b"_NET_WM_STATE_STICKY")?
        .reply()?
        .atom;
    connection.change_property32(
        PropMode::REPLACE,
        window,
        state,
        AtomEnum::ATOM,
        &[above, sticky],
    )?;
    connection.change_property8(
        PropMode::REPLACE,
        window,
        AtomEnum::WM_NAME,
        AtomEnum::STRING,
        b"Brazier Computer Use safety indicator",
    )?;
    Ok(())
}

fn overlay_position(connection: &RustConnection, root: u32, root_width: u16) -> (i16, i16) {
    let pointer = connection
        .query_pointer(root)
        .ok()
        .and_then(|cookie| cookie.reply().ok());
    let monitors = connection
        .randr_get_monitors(root, true)
        .ok()
        .and_then(|cookie| cookie.reply().ok());
    if let (Some(pointer), Some(monitors)) = (pointer, monitors)
        && let Some(monitor) = monitors.monitors.iter().find(|monitor| {
            let right = i32::from(monitor.x) + i32::from(monitor.width);
            let bottom = i32::from(monitor.y) + i32::from(monitor.height);
            i32::from(pointer.root_x) >= i32::from(monitor.x)
                && i32::from(pointer.root_x) < right
                && i32::from(pointer.root_y) >= i32::from(monitor.y)
                && i32::from(pointer.root_y) < bottom
        })
    {
        return (
            monitor.x + (monitor.width.saturating_sub(WIDTH) / 2) as i16,
            monitor.y.saturating_add(16),
        );
    }
    (((root_width.saturating_sub(WIDTH)) / 2) as i16, 16)
}

fn restack(connection: &RustConnection, window: u32, x: i16, y: i16) -> Result<()> {
    connection
        .configure_window(
            window,
            &ConfigureWindowAux::new()
                .x(i32::from(x))
                .y(i32::from(y))
                .stack_mode(StackMode::ABOVE),
        )?
        .check()
        .context("keep X11 safety overlay above other windows")
}

fn draw(connection: &RustConnection, window: u32, gc: u32) -> Result<()> {
    connection.change_gc(gc, &ChangeGCAux::new().foreground(0x31140e))?;
    connection.poly_fill_rectangle(
        window,
        gc,
        &[Rectangle {
            x: 0,
            y: 0,
            width: WIDTH,
            height: HEIGHT,
        }],
    )?;
    connection.change_gc(gc, &ChangeGCAux::new().foreground(0xff8c69))?;
    connection.poly_fill_rectangle(
        window,
        gc,
        &[
            Rectangle {
                x: 0,
                y: 0,
                width: WIDTH,
                height: 2,
            },
            Rectangle {
                x: 0,
                y: (HEIGHT - 2) as i16,
                width: WIDTH,
                height: 2,
            },
            Rectangle {
                x: 0,
                y: 0,
                width: 2,
                height: HEIGHT,
            },
            Rectangle {
                x: (WIDTH - 2) as i16,
                y: 0,
                width: 2,
                height: HEIGHT,
            },
        ],
    )?;
    connection.change_gc(gc, &ChangeGCAux::new().foreground(0xffad91))?;
    connection.image_text8(window, gc, 20, 33, b"Computer Use active")?;
    connection.change_gc(gc, &ChangeGCAux::new().foreground(0xffffff))?;
    connection.image_text8(window, gc, 248, 33, b"Esc to stop")?;
    Ok(())
}
