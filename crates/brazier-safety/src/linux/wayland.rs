use std::{
    io::{Seek as _, SeekFrom, Write as _},
    os::fd::AsFd as _,
    sync::mpsc::{self, SyncSender},
    thread,
    time::Duration,
};

use anyhow::{Context as _, Result, bail};
use wayland_client::{
    Connection, Dispatch, QueueHandle, delegate_noop,
    globals::{GlobalListContents, registry_queue_init},
    protocol::{wl_buffer, wl_compositor, wl_region, wl_registry, wl_shm, wl_shm_pool, wl_surface},
};
use wayland_protocols_wlr::layer_shell::v1::client::{
    zwlr_layer_shell_v1::{self, Layer},
    zwlr_layer_surface_v1::{self, Anchor, KeyboardInteractivity},
};

const WIDTH: u32 = 520;
const HEIGHT: u32 = 54;

pub struct Overlay {
    thread: thread::JoinHandle<Result<()>>,
}

impl Overlay {
    pub fn open() -> Result<Self> {
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let thread = thread::Builder::new()
            .name("brazier-wayland-safety-overlay".to_owned())
            .spawn(move || run_overlay(ready_tx))
            .context("spawn Wayland safety overlay thread")?;
        match ready_rx.recv_timeout(Duration::from_secs(8)) {
            Ok(Ok(())) => Ok(Self { thread }),
            Ok(Err(message)) => {
                let _ = thread.join();
                bail!(message)
            }
            Err(_) => bail!("timed out waiting for the Wayland safety overlay to become visible"),
        }
    }

    pub async fn wait(self) -> Result<()> {
        tokio::task::spawn_blocking(move || self.thread.join())
            .await
            .context("join Wayland safety overlay watcher")?
            .map_err(|_| anyhow::anyhow!("Wayland safety overlay thread panicked"))?
    }
}

struct State {
    surface: wl_surface::WlSurface,
    buffer: wl_buffer::WlBuffer,
    ready: Option<SyncSender<Result<(), String>>>,
    running: bool,
}

fn run_overlay(ready: SyncSender<Result<(), String>>) -> Result<()> {
    let result = run_overlay_inner(ready.clone());
    if let Err(error) = &result {
        let _ = ready.try_send(Err(format!("{error:#}")));
    }
    result
}

fn run_overlay_inner(ready: SyncSender<Result<(), String>>) -> Result<()> {
    let connection = Connection::connect_to_env().context("connect to the Wayland compositor")?;
    let (globals, mut events) = registry_queue_init::<State>(&connection)
        .context("read Wayland compositor capabilities")?;
    let queue = events.handle();
    let compositor: wl_compositor::WlCompositor = globals
        .bind(&queue, 1..=4, ())
        .context("Wayland compositor does not expose wl_compositor")?;
    let shm: wl_shm::WlShm = globals
        .bind(&queue, 1..=1, ())
        .context("Wayland compositor does not expose wl_shm")?;
    let layer_shell: zwlr_layer_shell_v1::ZwlrLayerShellV1 = globals
        .bind(&queue, 1..=4, ())
        .context(
            "the Wayland compositor does not support wlr-layer-shell; a security overlay cannot be guaranteed",
        )?;

    let surface = compositor.create_surface(&queue, ());
    surface.set_buffer_scale(1);
    let input_region = compositor.create_region(&queue, ());
    surface.set_input_region(Some(&input_region));
    input_region.destroy();

    let layer_surface = layer_shell.get_layer_surface(
        &surface,
        None,
        Layer::Overlay,
        "brazier-computer-use-safety".to_owned(),
        &queue,
        (),
    );
    layer_surface.set_size(WIDTH, HEIGHT);
    layer_surface.set_anchor(Anchor::Top);
    layer_surface.set_margin(16, 0, 0, 0);
    layer_surface.set_exclusive_zone(0);
    layer_surface.set_keyboard_interactivity(KeyboardInteractivity::None);

    let buffer = make_buffer(&shm, &queue)?;
    let mut state = State {
        surface,
        buffer,
        ready: Some(ready),
        running: true,
    };
    // Layer-shell requires a bufferless initial commit followed by configure.
    state.surface.commit();
    connection.flush().context("send Wayland overlay request")?;
    while state.running {
        events
            .blocking_dispatch(&mut state)
            .context("Wayland safety overlay disconnected")?;
    }
    bail!("the compositor closed the Wayland safety overlay")
}

fn make_buffer(shm: &wl_shm::WlShm, queue: &QueueHandle<State>) -> Result<wl_buffer::WlBuffer> {
    let mut file = tempfile::tempfile().context("create safety overlay pixel buffer")?;
    let pixels = draw_banner();
    file.set_len(pixels.len() as u64)
        .context("size safety overlay pixel buffer")?;
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&pixels)
        .context("draw safety overlay pixel buffer")?;
    file.flush()?;
    let pool = shm.create_pool(file.as_fd(), pixels.len() as i32, queue, ());
    let buffer = pool.create_buffer(
        0,
        WIDTH as i32,
        HEIGHT as i32,
        (WIDTH * 4) as i32,
        wl_shm::Format::Argb8888,
        queue,
        (),
    );
    pool.destroy();
    Ok(buffer)
}

fn draw_banner() -> Vec<u8> {
    let mut pixels = vec![0_u8; (WIDTH * HEIGHT * 4) as usize];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let radius = 12_i32;
            let inside = if x < radius as u32 && y < radius as u32 {
                circle(x as i32 - radius, y as i32 - radius, radius)
            } else if x >= WIDTH - radius as u32 && y < radius as u32 {
                circle(
                    x as i32 - (WIDTH as i32 - radius - 1),
                    y as i32 - radius,
                    radius,
                )
            } else if x < radius as u32 && y >= HEIGHT - radius as u32 {
                circle(
                    x as i32 - radius,
                    y as i32 - (HEIGHT as i32 - radius - 1),
                    radius,
                )
            } else if x >= WIDTH - radius as u32 && y >= HEIGHT - radius as u32 {
                circle(
                    x as i32 - (WIDTH as i32 - radius - 1),
                    y as i32 - (HEIGHT as i32 - radius - 1),
                    radius,
                )
            } else {
                true
            };
            if inside {
                let border = x <= 1 || y <= 1 || x >= WIDTH - 2 || y >= HEIGHT - 2;
                set_pixel(
                    &mut pixels,
                    x,
                    y,
                    if border {
                        (255, 140, 105, 255)
                    } else {
                        (49, 20, 14, 246)
                    },
                );
            }
        }
    }
    draw_text(
        &mut pixels,
        21,
        18,
        "Computer Use active",
        (255, 173, 145, 255),
    );
    draw_text(
        &mut pixels,
        245,
        18,
        "Ctrl+Shift+Esc to stop",
        (255, 255, 255, 255),
    );
    pixels
}

fn circle(x: i32, y: i32, radius: i32) -> bool {
    x * x + y * y <= radius * radius
}

fn set_pixel(pixels: &mut [u8], x: u32, y: u32, (r, g, b, a): (u8, u8, u8, u8)) {
    let offset = ((y * WIDTH + x) * 4) as usize;
    // wl_shm ARGB8888 is native-endian; on supported Linux architectures its
    // byte order in memory is BGRA.
    pixels[offset..offset + 4].copy_from_slice(&[b, g, r, a]);
}

fn draw_text(pixels: &mut [u8], mut x: u32, y: u32, text: &str, color: (u8, u8, u8, u8)) {
    for character in text.chars() {
        if character == ' ' {
            x += 8;
            continue;
        }
        let rows = glyph(character.to_ascii_uppercase());
        for (row, bits) in rows.into_iter().enumerate() {
            for column in 0..5 {
                if bits & (1 << (4 - column)) != 0 {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            set_pixel(pixels, x + column * 2 + dx, y + row as u32 * 2 + dy, color);
                        }
                    }
                }
            }
        }
        x += 12;
    }
}

fn glyph(character: char) -> [u8; 7] {
    match character {
        'A' => [0x0e, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11],
        'C' => [0x0e, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0e],
        'E' => [0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x1f],
        'F' => [0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x10],
        'H' => [0x11, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11],
        'I' => [0x1f, 0x04, 0x04, 0x04, 0x04, 0x04, 0x1f],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1f],
        'M' => [0x11, 0x1b, 0x15, 0x15, 0x11, 0x11, 0x11],
        'O' => [0x0e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e],
        'P' => [0x1e, 0x11, 0x11, 0x1e, 0x10, 0x10, 0x10],
        '+' => [0x00, 0x04, 0x04, 0x1f, 0x04, 0x04, 0x00],
        'R' => [0x1e, 0x11, 0x11, 0x1e, 0x14, 0x12, 0x11],
        'S' => [0x0f, 0x10, 0x10, 0x0e, 0x01, 0x01, 0x1e],
        'T' => [0x1f, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0a, 0x04],
        _ => [0, 0, 0, 0, 0, 0, 0],
    }
}

impl Dispatch<wl_registry::WlRegistry, GlobalListContents> for State {
    fn event(
        _: &mut Self,
        _: &wl_registry::WlRegistry,
        _: wl_registry::Event,
        _: &GlobalListContents,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

delegate_noop!(State: ignore wl_compositor::WlCompositor);
delegate_noop!(State: ignore wl_surface::WlSurface);
delegate_noop!(State: ignore wl_region::WlRegion);
delegate_noop!(State: ignore wl_shm::WlShm);
delegate_noop!(State: ignore wl_shm_pool::WlShmPool);
delegate_noop!(State: ignore wl_buffer::WlBuffer);
delegate_noop!(State: ignore zwlr_layer_shell_v1::ZwlrLayerShellV1);

impl Dispatch<zwlr_layer_surface_v1::ZwlrLayerSurfaceV1, ()> for State {
    fn event(
        state: &mut Self,
        layer_surface: &zwlr_layer_surface_v1::ZwlrLayerSurfaceV1,
        event: zwlr_layer_surface_v1::Event,
        _: &(),
        connection: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwlr_layer_surface_v1::Event::Configure { serial, .. } => {
                layer_surface.ack_configure(serial);
                state.surface.attach(Some(&state.buffer), 0, 0);
                state
                    .surface
                    .damage_buffer(0, 0, WIDTH as i32, HEIGHT as i32);
                state.surface.commit();
                let _ = connection.flush();
                if let Some(ready) = state.ready.take() {
                    let _ = ready.send(Ok(()));
                }
            }
            zwlr_layer_surface_v1::Event::Closed => state.running = false,
            _ => {}
        }
    }
}
