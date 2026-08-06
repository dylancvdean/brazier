//! Direct X11 capture and XTEST input. No xdotool or ImageMagick process.

use x11rb::{
    connection::Connection,
    protocol::{
        xproto::{
            BUTTON_PRESS_EVENT, BUTTON_RELEASE_EVENT, ConnectionExt as _, ImageFormat,
            MOTION_NOTIFY_EVENT,
        },
        xtest::ConnectionExt as _,
    },
    rust_connection::RustConnection,
};

fn connection() -> Result<(RustConnection, usize), String> {
    RustConnection::connect(None).map_err(|e| format!("connect to X11: {e}"))
}
fn root(connection: &RustConnection, screen: usize) -> Result<(u32, u16, u16), String> {
    let screen = connection
        .setup()
        .roots
        .get(screen)
        .ok_or("X11 screen is unavailable")?;
    Ok((screen.root, screen.width_in_pixels, screen.height_in_pixels))
}
fn coordinate(value: f64) -> i16 {
    value.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16
}

pub fn screenshot() -> Result<Vec<u8>, String> {
    let (connection, screen) = connection()?;
    let (root, width, height) = root(&connection, screen)?;
    let image = connection
        .get_image(ImageFormat::Z_PIXMAP, root, 0, 0, width, height, u32::MAX)
        .map_err(|e| e.to_string())?
        .reply()
        .map_err(|e| e.to_string())?;
    let pixels = usize::from(width) * usize::from(height);
    let stride = image.data.len() / pixels;
    if !matches!(stride, 3 | 4) {
        return Err(format!("unsupported X11 pixel stride {stride}"));
    }
    let mut rgb = Vec::with_capacity(pixels * 3);
    for pixel in image.data.chunks_exact(stride) {
        rgb.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
    }
    let mut png = Vec::new();
    let mut encoder = png::Encoder::new(&mut png, u32::from(width), u32::from(height));
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .map_err(|e| e.to_string())?
        .write_image_data(&rgb)
        .map_err(|e| e.to_string())?;
    Ok(png)
}

fn motion(connection: &RustConnection, root: u32, x: f64, y: f64) -> Result<(), String> {
    connection
        .xtest_fake_input(
            MOTION_NOTIFY_EVENT,
            0,
            0,
            root,
            coordinate(x),
            coordinate(y),
            0,
        )
        .map_err(|e| e.to_string())?
        .check()
        .map_err(|e| e.to_string())?;
    connection.flush().map_err(|e| e.to_string())
}
fn button(connection: &RustConnection, root: u32, button: u8, pressed: bool) -> Result<(), String> {
    connection
        .xtest_fake_input(
            if pressed {
                BUTTON_PRESS_EVENT
            } else {
                BUTTON_RELEASE_EVENT
            },
            button,
            0,
            root,
            0,
            0,
            0,
        )
        .map_err(|e| e.to_string())?
        .check()
        .map_err(|e| e.to_string())?;
    connection.flush().map_err(|e| e.to_string())
}
pub fn click(x: f64, y: f64, button_number: u8, count: usize) -> Result<(), String> {
    let (connection, screen) = connection()?;
    let (root, _, _) = root(&connection, screen)?;
    motion(&connection, root, x, y)?;
    for _ in 0..count {
        button(&connection, root, button_number, true)?;
        button(&connection, root, button_number, false)?;
    }
    Ok(())
}
pub fn move_to(x: f64, y: f64) -> Result<(), String> {
    let (c, s) = connection()?;
    let (r, _, _) = root(&c, s)?;
    motion(&c, r, x, y)
}
pub fn drag(start_x: f64, start_y: f64, end_x: f64, end_y: f64) -> Result<(), String> {
    let (c, s) = connection()?;
    let (r, _, _) = root(&c, s)?;
    motion(&c, r, start_x, start_y)?;
    button(&c, r, 1, true)?;
    motion(&c, r, end_x, end_y)?;
    button(&c, r, 1, false)
}
pub fn scroll(delta_y: f64) -> Result<(), String> {
    let (c, s) = connection()?;
    let (r, _, _) = root(&c, s)?;
    let button_number = if delta_y < 0.0 { 4 } else { 5 };
    for _ in 0..delta_y.abs().round() as usize {
        button(&c, r, button_number, true)?;
        button(&c, r, button_number, false)?;
    }
    Ok(())
}

fn keycode(connection: &RustConnection, keysym: u32) -> Result<u8, String> {
    let setup = connection.setup();
    let first = setup.min_keycode;
    let count = setup.max_keycode - first + 1;
    let mapping = connection
        .get_keyboard_mapping(first, count)
        .map_err(|e| e.to_string())?
        .reply()
        .map_err(|e| e.to_string())?;
    mapping
        .keysyms
        .chunks(usize::from(mapping.keysyms_per_keycode))
        .position(|symbols| symbols.contains(&keysym))
        .map(|index| first + index as u8)
        .ok_or_else(|| format!("X11 layout has no key for keysym {keysym:#x}"))
}
pub fn key(keysym: u32) -> Result<(), String> {
    let (c, s) = connection()?;
    let (r, _, _) = root(&c, s)?;
    let code = keycode(&c, keysym)?;
    c.xtest_fake_input(2, code, 0, r, 0, 0, 0)
        .map_err(|e| e.to_string())?
        .check()
        .map_err(|e| e.to_string())?;
    c.xtest_fake_input(3, code, 0, r, 0, 0, 0)
        .map_err(|e| e.to_string())?
        .check()
        .map_err(|e| e.to_string())?;
    c.flush().map_err(|e| e.to_string())
}
pub fn type_text(text: &str) -> Result<(), String> {
    for ch in text.chars() {
        key(ch as u32)?;
    }
    Ok(())
}
