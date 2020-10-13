use rlua::prelude::*;
use std::convert::TryFrom;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use vulkano_win::VkSurfaceBuild;

#[derive(PartialEq, Eq, Hash, Clone)]
pub enum WindowEvent {
    Resized,
}

impl<'a, T> TryFrom<&winit::event::Event<'a, T>> for WindowEvent {
    type Error = String;

    fn try_from(ev: &winit::event::Event<'a, T>) -> Result<WindowEvent, String> {
        match ev {
            winit::event::Event::WindowEvent {
                event: winit::event::WindowEvent::Resized(_),
                ..
            } => Ok(WindowEvent::Resized),
            _ => Err("unsupported event type".to_owned()),
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub enum ProgramEvent {
    CloseRequested,
}

impl<'a, T> TryFrom<&winit::event::Event<'a, T>> for ProgramEvent {
    type Error = String;

    fn try_from(ev: &winit::event::Event<'a, T>) -> Result<ProgramEvent, String> {
        match ev {
            winit::event::Event::WindowEvent {
                event: winit::event::WindowEvent::CloseRequested,
                ..
            } => Ok(ProgramEvent::CloseRequested),
            _ => Err("unsupported event type".to_owned()),
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub enum InputEvent {
    ButtonPressed(String),
    ButtonReleased(String),
}

fn map_keycode(code: winit::event::VirtualKeyCode) -> Result<&'static str, &'static str> {
    match code {
        winit::event::VirtualKeyCode::A => Ok("a"),
        winit::event::VirtualKeyCode::B => Ok("b"),
        winit::event::VirtualKeyCode::C => Ok("c"),
        winit::event::VirtualKeyCode::D => Ok("d"),
        winit::event::VirtualKeyCode::E => Ok("e"),
        winit::event::VirtualKeyCode::F => Ok("f"),
        winit::event::VirtualKeyCode::G => Ok("g"),
        winit::event::VirtualKeyCode::H => Ok("h"),
        winit::event::VirtualKeyCode::J => Ok("j"),
        winit::event::VirtualKeyCode::I => Ok("i"),
        winit::event::VirtualKeyCode::K => Ok("k"),
        winit::event::VirtualKeyCode::L => Ok("l"),
        winit::event::VirtualKeyCode::M => Ok("m"),
        winit::event::VirtualKeyCode::N => Ok("n"),
        winit::event::VirtualKeyCode::O => Ok("o"),
        winit::event::VirtualKeyCode::P => Ok("p"),
        winit::event::VirtualKeyCode::Q => Ok("q"),
        winit::event::VirtualKeyCode::R => Ok("r"),
        winit::event::VirtualKeyCode::S => Ok("s"),
        winit::event::VirtualKeyCode::T => Ok("t"),
        winit::event::VirtualKeyCode::U => Ok("u"),
        winit::event::VirtualKeyCode::V => Ok("v"),
        winit::event::VirtualKeyCode::W => Ok("w"),
        winit::event::VirtualKeyCode::X => Ok("x"),
        winit::event::VirtualKeyCode::Y => Ok("y"),
        winit::event::VirtualKeyCode::Z => Ok("z"),
        winit::event::VirtualKeyCode::Down => Ok("down"),
        winit::event::VirtualKeyCode::Up => Ok("up"),
        winit::event::VirtualKeyCode::Left => Ok("left"),
        winit::event::VirtualKeyCode::Right => Ok("right"),
        winit::event::VirtualKeyCode::Key0 => Ok("0"),
        winit::event::VirtualKeyCode::Key1 => Ok("1"),
        winit::event::VirtualKeyCode::Key2 => Ok("2"),
        winit::event::VirtualKeyCode::Key3 => Ok("3"),
        winit::event::VirtualKeyCode::Key4 => Ok("4"),
        winit::event::VirtualKeyCode::Key5 => Ok("5"),
        winit::event::VirtualKeyCode::Key6 => Ok("6"),
        winit::event::VirtualKeyCode::Key7 => Ok("7"),
        winit::event::VirtualKeyCode::Key8 => Ok("8"),
        winit::event::VirtualKeyCode::Key9 => Ok("9"),
        winit::event::VirtualKeyCode::Space => Ok("space"),
        winit::event::VirtualKeyCode::Tab => Ok("tab"),
        winit::event::VirtualKeyCode::Capital => Ok("shift"),
        _ => Err("Cannot convert!"),
    }
}

impl<'a, T> TryFrom<&winit::event::Event<'a, T>> for InputEvent {
    type Error = String;

    fn try_from(ev: &winit::event::Event<'a, T>) -> Result<InputEvent, String> {
        match ev {
            winit::event::Event::WindowEvent {
                window_id: _,
                event:
                    winit::event::WindowEvent::KeyboardInput {
                        device_id: _,
                        input,
                        is_synthetic: _,
                    },
            } => {
                let keycode = input.virtual_keycode.ok_or("No keycode present")?;
                match input.state {
                    winit::event::ElementState::Pressed => {
                        Ok(InputEvent::ButtonPressed(map_keycode(keycode)?.to_owned()))
                    }
                    winit::event::ElementState::Released => {
                        Ok(InputEvent::ButtonReleased(map_keycode(keycode)?.to_owned()))
                    }
                }
            }
            _ => Err("unsupported event type".to_owned()),
        }
    }
}

impl<'lua> ToLua<'lua> for InputEvent {
    fn to_lua(self, context: LuaContext<'lua>) -> Result<LuaValue<'lua>, LuaError> {
        let ev = context.create_table()?;
        match self {
            InputEvent::ButtonPressed(s) => {
                ev.set("type", "ButtonPressed")?;
                ev.set("button", s)?;
            }
            InputEvent::ButtonReleased(s) => {
                ev.set("type", "ButtonReleased")?;
                ev.set("button", s)?;
            }
        }
        Ok(LuaValue::Table(ev))
    }
}

pub struct WindowSystem {
    event_loop: winit::event_loop::EventLoop<()>,
    window_send: Sender<WindowEvent>,
    window_recieve: Option<Receiver<WindowEvent>>,
    input_send: Sender<InputEvent>,
    input_recieve: Option<Receiver<InputEvent>>,
    program_send: Sender<ProgramEvent>,
    program_recieve: Option<Receiver<ProgramEvent>>,
}

impl WindowSystem {
    pub fn new() -> WindowSystem {
        let event_loop = winit::event_loop::EventLoop::new();
        let (window_send, window_recieve) = channel();
        let (input_send, input_recieve) = channel();
        let (program_send, program_recieve) = channel();
        WindowSystem {
            event_loop,
            window_send,
            window_recieve: Some(window_recieve),
            input_send,
            input_recieve: Some(input_recieve),
            program_send,
            program_recieve: Some(program_recieve),
        }
    }

    pub fn input_event_reciever(&mut self) -> Option<Receiver<InputEvent>> {
        self.input_recieve.take()
    }

    pub fn window_event_sender(&self) -> Sender<WindowEvent> {
        self.window_send.clone()
    }

    pub fn window_event_reciever(&mut self) -> Option<Receiver<WindowEvent>> {
        self.window_recieve.take()
    }

    pub fn program_event_reciever(&mut self) -> Option<Receiver<ProgramEvent>> {
        self.program_recieve.take()
    }

    pub fn run_event_loop(self, ecs_thread: std::thread::JoinHandle<()>) -> ! {
        let mut thread = Some(ecs_thread);
        let mut should_close = false;
        let program_send = self.program_send;
        let input_send = self.input_send;
        let window_send = self.window_send;

        self.event_loop.run(move |ev, _, control_flow| {
            if let Ok(event) = WinitEvent::try_from(ev) {
                match send_event(event, &program_send, &input_send, &window_send) {
                    Ok(Some(ev)) => should_close = true,
                    Ok(None) => (),
                    Err(e) => error!("Failed to send event: {}", e),
                }
            }

            if should_close {
                if let Some(thread) = thread.take() {
                    thread.join().expect("Failed to stop ecs thread");
                }
                *control_flow = winit::event_loop::ControlFlow::Exit;
                return;
            }
            *control_flow = winit::event_loop::ControlFlow::Wait;
        })
    }

    pub fn surface(
        &self,
        instance: std::sync::Arc<vulkano::instance::Instance>,
    ) -> Result<Arc<vulkano::swapchain::Surface<winit::window::Window>>, vulkano_win::CreationError>
    {
        winit::window::WindowBuilder::new().build_vk_surface(&self.event_loop, instance)
    }
}

fn send_event(
    event: WinitEvent,
    program_send: &Sender<crate::windowing::ProgramEvent>,
    input_send: &Sender<crate::windowing::InputEvent>,
    window_send: &Sender<crate::windowing::WindowEvent>,
) -> Result<Option<ProgramEvent>, String> {
    match event {
        WinitEvent::ProgramEvent(program_event) => program_send
            .send(program_event.clone())
            .map_err(|y| format!("Failed to send chan, {}", y))
            .map(|x| Some(program_event)),
        WinitEvent::InputEvent(input_event) => input_send
            .send(input_event)
            .map_err(|y| format!("Failed to send chan, {}", y))
            .map(|x| None),
        WinitEvent::WindowEvent(window_event) => window_send
            .send(window_event)
            .map_err(|y| format!("Failed to send chan, {}", y))
            .map(|x| None),
    }
}

enum WinitEvent {
    ProgramEvent(ProgramEvent),
    WindowEvent(WindowEvent),
    InputEvent(InputEvent),
}

impl<'a, T> TryFrom<winit::event::Event<'a, T>> for WinitEvent {
    type Error = String;

    fn try_from(ev: winit::event::Event<'a, T>) -> Result<WinitEvent, String> {
        if let Ok(program_event) = ProgramEvent::try_from(&ev) {
            return Ok(WinitEvent::ProgramEvent(program_event));
        }

        if let Ok(window_event) = WindowEvent::try_from(&ev) {
            return Ok(WinitEvent::WindowEvent(window_event));
        }

        if let Ok(input_event) = InputEvent::try_from(&ev) {
            return Ok(WinitEvent::InputEvent(input_event));
        }

        Err("Unsupported event type".to_owned())
    }
}
