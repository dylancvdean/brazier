use std::{
    collections::HashMap,
    ffi::CString,
    fs::{self, File, OpenOptions},
    io::{ErrorKind, Write as _},
    mem::{MaybeUninit, size_of},
    os::{
        fd::{AsRawFd as _, FromRawFd as _},
        unix::fs::OpenOptionsExt as _,
    },
    path::{Path, PathBuf},
};

use anyhow::{Context as _, Result, bail};

const INPUT_DIRECTORY: &str = "/dev/input";
const EV_SYN: u16 = 0;
const EV_KEY: u16 = 1;
const SYN_DROPPED: u16 = 3;
const KEY_ESC: u16 = 1;
const KEY_LEFTCTRL: u16 = 29;
const KEY_LEFTSHIFT: u16 = 42;
const KEY_RIGHTSHIFT: u16 = 54;
const KEY_RIGHTCTRL: u16 = 97;
const KEY_MAX: usize = 0x2ff;

#[repr(C)]
#[derive(Clone, Copy)]
struct InputEvent {
    time: libc::timeval,
    kind: u16,
    code: u16,
    value: i32,
}

struct Keyboard {
    file: File,
    ctrl: u8,
    shift: u8,
    escape: bool,
}

struct InputDirectory(File);

impl InputDirectory {
    fn open() -> Result<Self> {
        let descriptor = unsafe { libc::inotify_init1(libc::IN_NONBLOCK | libc::IN_CLOEXEC) };
        if descriptor < 0 {
            return Err(std::io::Error::last_os_error()).context("create input-device watcher");
        }
        let file = unsafe { File::from_raw_fd(descriptor) };
        let path = CString::new(INPUT_DIRECTORY).expect("fixed input path contains no nul byte");
        let mask = libc::IN_CREATE
            | libc::IN_MOVED_TO
            | libc::IN_DELETE
            | libc::IN_MOVED_FROM
            | libc::IN_ATTRIB;
        if unsafe { libc::inotify_add_watch(file.as_raw_fd(), path.as_ptr(), mask) } < 0 {
            return Err(std::io::Error::last_os_error()).context("watch /dev/input changes");
        }
        Ok(Self(file))
    }

    fn drain(&self) -> Result<()> {
        let mut buffer = [0_u8; 4096];
        loop {
            let read =
                unsafe { libc::read(self.0.as_raw_fd(), buffer.as_mut_ptr().cast(), buffer.len()) };
            if read > 0 {
                let read = read as usize;
                let mut offset = 0;
                while offset < read {
                    const HEADER: usize = 16;
                    if read - offset < HEADER {
                        bail!("received a partial input-device notification")
                    }
                    let mask = u32::from_ne_bytes(
                        buffer[offset + 4..offset + 8]
                            .try_into()
                            .expect("inotify mask has a fixed size"),
                    );
                    let name_length = u32::from_ne_bytes(
                        buffer[offset + 12..offset + 16]
                            .try_into()
                            .expect("inotify name length has a fixed size"),
                    ) as usize;
                    let event_length = HEADER
                        .checked_add(name_length)
                        .context("input-device notification length overflow")?;
                    if event_length > read - offset {
                        bail!("received a truncated input-device notification")
                    }
                    if mask & (libc::IN_IGNORED | libc::IN_UNMOUNT) != 0 {
                        bail!(
                            "the input-device directory watch was removed; computer use cannot continue safely"
                        )
                    }
                    offset += event_length;
                }
                continue;
            }
            if read == 0 {
                return Ok(());
            }
            let error = std::io::Error::last_os_error();
            if error.kind() == ErrorKind::WouldBlock {
                return Ok(());
            }
            if error.kind() == ErrorKind::Interrupted {
                continue;
            }
            return Err(error).context("read input-device changes");
        }
    }
}

pub fn run() -> Result<()> {
    let probe = std::env::args_os().any(|argument| argument == "--probe");
    harden_process()?;

    let input_directory = InputDirectory::open()?;
    let mut keyboards = HashMap::new();
    let emergency_already_held = scan_keyboards(&mut keyboards)?;
    if keyboards.is_empty() {
        bail!(
            "no readable keyboard input devices; install the Brazier privileged safety fallback from Settings > Computer Use permissions"
        )
    }

    println!("READY {}", env!("CARGO_PKG_VERSION"));
    std::io::stdout()
        .flush()
        .context("flush input guard readiness")?;
    if probe {
        return Ok(());
    }
    if emergency_already_held {
        println!("ESC");
        std::io::stdout().flush().context("flush emergency stop")?;
        return Ok(());
    }

    watch(&mut keyboards, input_directory)
}

fn harden_process() -> Result<()> {
    // Key state is not secret data we retain, but a setgid safety process
    // should not produce core dumps or be ptraceable by sibling processes.
    let parent = unsafe { libc::getppid() };
    for (operation, result) in [
        ("bind input guard lifetime to its parent", unsafe {
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM, 0, 0, 0)
        }),
        ("disable input guard core dumps", unsafe {
            libc::prctl(libc::PR_SET_DUMPABLE, 0, 0, 0, 0)
        }),
        ("prevent input guard privilege escalation", unsafe {
            libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
        }),
    ] {
        if result != 0 {
            return Err(std::io::Error::last_os_error()).context(operation);
        }
    }
    if unsafe { libc::getppid() } != parent {
        bail!("input guard parent exited during startup")
    }
    Ok(())
}

fn watch(
    keyboards: &mut HashMap<PathBuf, Keyboard>,
    input_directory: InputDirectory,
) -> Result<()> {
    loop {
        let paths: Vec<PathBuf> = keyboards.keys().cloned().collect();
        let mut poll_fds = vec![libc::pollfd {
            fd: input_directory.0.as_raw_fd(),
            events: libc::POLLIN | libc::POLLERR | libc::POLLHUP,
            revents: 0,
        }];
        poll_fds.extend(paths.iter().map(|path| libc::pollfd {
            fd: keyboards[path].file.as_raw_fd(),
            events: libc::POLLIN | libc::POLLERR | libc::POLLHUP,
            revents: 0,
        }));
        let result = unsafe { libc::poll(poll_fds.as_mut_ptr(), poll_fds.len() as _, 1000) };
        if result < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == ErrorKind::Interrupted {
                continue;
            }
            return Err(error).context("watch keyboard input devices");
        }

        if poll_fds[0].revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
            bail!("the input-device directory watcher stopped; computer use cannot continue safely")
        }
        if poll_fds[0].revents & libc::POLLIN != 0 {
            input_directory.drain()?;
        }

        let mut disconnected = Vec::new();
        for (path, descriptor) in paths.iter().zip(&poll_fds[1..]) {
            if descriptor.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
                disconnected.push(path.clone());
                continue;
            }
            if descriptor.revents & libc::POLLIN != 0 && read_events(path, keyboards)? {
                println!("ESC");
                std::io::stdout().flush().context("flush emergency stop")?;
                return Ok(());
            }
        }
        for path in disconnected {
            keyboards.remove(&path);
        }

        if scan_keyboards(keyboards)? {
            println!("ESC");
            std::io::stdout().flush().context("flush emergency stop")?;
            return Ok(());
        }
        if keyboards.is_empty() {
            bail!("all readable keyboard devices disappeared; computer use cannot continue safely")
        }
    }
}

fn read_events(path: &Path, keyboards: &mut HashMap<PathBuf, Keyboard>) -> Result<bool> {
    loop {
        let mut event = MaybeUninit::<InputEvent>::uninit();
        let read = unsafe {
            libc::read(
                keyboards[path].file.as_raw_fd(),
                event.as_mut_ptr().cast(),
                size_of::<InputEvent>(),
            )
        };
        if read < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == ErrorKind::WouldBlock {
                return Ok(false);
            }
            return Err(error).with_context(|| format!("read {}", path.display()));
        }
        if read == 0 {
            return Ok(false);
        }
        if read as usize != size_of::<InputEvent>() {
            bail!("received a partial input event from {}", path.display())
        }
        let event = unsafe { event.assume_init() };
        if event.kind == EV_SYN && event.code == SYN_DROPPED {
            bail!("keyboard events were dropped; computer use cannot continue safely")
        }
        if event.kind != EV_KEY {
            continue;
        }

        let pressed = event.value != 0;
        match event.code {
            KEY_LEFTCTRL => {
                set_modifier(
                    &mut keyboards
                        .get_mut(path)
                        .context("keyboard disappeared while processing input")?
                        .ctrl,
                    0b01,
                    pressed,
                );
            }
            KEY_RIGHTCTRL => {
                set_modifier(
                    &mut keyboards
                        .get_mut(path)
                        .context("keyboard disappeared while processing input")?
                        .ctrl,
                    0b10,
                    pressed,
                );
            }
            KEY_LEFTSHIFT => {
                set_modifier(
                    &mut keyboards
                        .get_mut(path)
                        .context("keyboard disappeared while processing input")?
                        .shift,
                    0b01,
                    pressed,
                );
            }
            KEY_RIGHTSHIFT => {
                set_modifier(
                    &mut keyboards
                        .get_mut(path)
                        .context("keyboard disappeared while processing input")?
                        .shift,
                    0b10,
                    pressed,
                );
            }
            KEY_ESC => {
                keyboards
                    .get_mut(path)
                    .context("keyboard disappeared while processing input")?
                    .escape = pressed;
            }
            _ => {}
        }
        if event.value != 0 && emergency_held(keyboards.values()) {
            return Ok(true);
        }
    }
}

fn set_modifier(state: &mut u8, flag: u8, pressed: bool) {
    if pressed {
        *state |= flag;
    } else {
        *state &= !flag;
    }
}

fn emergency_held<'a>(states: impl Iterator<Item = &'a Keyboard>) -> bool {
    emergency_state(states.map(|device| (device.ctrl != 0, device.shift != 0, device.escape)))
}

fn emergency_state(states: impl Iterator<Item = (bool, bool, bool)>) -> bool {
    let (mut ctrl, mut shift, mut escape) = (false, false, false);
    for (device_ctrl, device_shift, device_escape) in states {
        ctrl |= device_ctrl;
        shift |= device_shift;
        escape |= device_escape;
    }
    ctrl && shift && escape
}

fn scan_keyboards(keyboards: &mut HashMap<PathBuf, Keyboard>) -> Result<bool> {
    let entries = fs::read_dir(INPUT_DIRECTORY).context("open /dev/input")?;
    for entry in entries {
        let entry = entry.context("enumerate /dev/input")?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.strip_prefix("event").is_some_and(|suffix| {
            !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
        }) {
            continue;
        }
        let path = entry.path();
        if keyboards.contains_key(&path) {
            continue;
        }
        let Some(keyboard) = open_keyboard(&path)? else {
            continue;
        };
        keyboards.insert(path, keyboard);
    }
    Ok(emergency_held(keyboards.values()))
}

fn open_keyboard(path: &Path) -> Result<Option<Keyboard>> {
    let file = match OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
    {
        Ok(file) => file,
        Err(error)
            if matches!(
                error.kind(),
                ErrorKind::PermissionDenied | ErrorKind::NotFound
            ) =>
        {
            return Ok(None);
        }
        Err(error) => return Err(error).with_context(|| format!("open {}", path.display())),
    };
    let metadata = file
        .metadata()
        .with_context(|| format!("inspect {}", path.display()))?;
    use std::os::unix::fs::{FileTypeExt as _, MetadataExt as _};
    if !metadata.file_type().is_char_device() || libc::major(metadata.rdev()) != 13 {
        return Ok(None);
    }

    let mut key_bits = [0_u8; (KEY_MAX + 8) / 8];
    let request = ev_ioc_gbit(EV_KEY, key_bits.len());
    let result = unsafe { libc::ioctl(file.as_raw_fd(), request, key_bits.as_mut_ptr()) };
    if result < 0 {
        return Ok(None);
    }
    if !bit_is_set(&key_bits, KEY_ESC)
        || !(bit_is_set(&key_bits, KEY_LEFTCTRL) || bit_is_set(&key_bits, KEY_RIGHTCTRL))
        || !(bit_is_set(&key_bits, KEY_LEFTSHIFT) || bit_is_set(&key_bits, KEY_RIGHTSHIFT))
    {
        return Ok(None);
    }

    let mut pressed = [0_u8; (KEY_MAX + 8) / 8];
    let result = unsafe {
        libc::ioctl(
            file.as_raw_fd(),
            ev_ioc_gkey(pressed.len()),
            pressed.as_mut_ptr(),
        )
    };
    if result < 0 {
        return Ok(None);
    }
    let ctrl = u8::from(bit_is_set(&pressed, KEY_LEFTCTRL))
        | (u8::from(bit_is_set(&pressed, KEY_RIGHTCTRL)) << 1);
    let shift = u8::from(bit_is_set(&pressed, KEY_LEFTSHIFT))
        | (u8::from(bit_is_set(&pressed, KEY_RIGHTSHIFT)) << 1);
    let escape = bit_is_set(&pressed, KEY_ESC);
    Ok(Some(Keyboard {
        file,
        ctrl,
        shift,
        escape,
    }))
}

fn bit_is_set(bits: &[u8], bit: u16) -> bool {
    bits.get(bit as usize / 8)
        .is_some_and(|byte| byte & (1 << (bit % 8)) != 0)
}

fn ev_ioc_gbit(event_type: u16, length: usize) -> libc::c_ulong {
    const IOC_READ: libc::c_ulong = 2;
    (IOC_READ << 30)
        | ((length as libc::c_ulong) << 16)
        | ((b'E' as libc::c_ulong) << 8)
        | (0x20 + event_type as libc::c_ulong)
}

fn ev_ioc_gkey(length: usize) -> libc::c_ulong {
    const IOC_READ: libc::c_ulong = 2;
    (IOC_READ << 30) | ((length as libc::c_ulong) << 16) | ((b'E' as libc::c_ulong) << 8) | 0x18
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_bits_are_checked_safely() {
        let mut bits = [0_u8; 2];
        bits[1] = 0b0000_0100;
        assert!(bit_is_set(&bits, 10));
        assert!(!bit_is_set(&bits, 11));
        assert!(!bit_is_set(&bits, 40));
    }

    #[test]
    fn evdev_ioctl_uses_the_expected_linux_encoding() {
        assert_eq!(ev_ioc_gbit(EV_KEY, 96), 0x8060_4521);
        assert_eq!(ev_ioc_gkey(96), 0x8060_4518);
    }

    #[test]
    fn emergency_chord_requires_both_modifiers_and_an_escape_press() {
        assert!(emergency_state([(true, true, true)].into_iter()));
        assert!(emergency_state(
            [(true, false, false), (false, true, true)].into_iter()
        ));
        assert!(!emergency_state([(true, false, true)].into_iter()));
        assert!(!emergency_state([(true, true, false)].into_iter()));
    }

    #[test]
    fn left_and_right_modifiers_are_tracked_independently() {
        let mut state = 0;
        set_modifier(&mut state, 0b01, true);
        set_modifier(&mut state, 0b10, true);
        set_modifier(&mut state, 0b01, false);
        assert_eq!(state, 0b10);
    }
}
