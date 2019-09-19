use super::core::{MethodAdder, Scriptable};

#[derive(Debug)]
pub struct Position(f32, f32);

impl Position {
    pub fn get(&self) -> (f32, f32) {
        (self.0, self.1)
    }

    pub fn set(&mut self, (x, y): (f32, f32)) {
        self.0 = x;
        self.1 = y;
    }

    pub fn new(x: f32, y: f32) -> Position {
        Position(x, y)
    }
}

impl Scriptable for Position {
    fn add_methods<'lua, M: MethodAdder<'lua, Self>>(methods: &mut M) {
        methods.add_method_mut("set", |_context, this, (x, y)| Ok(this.set((x, y))));

        methods.add_method("get", |_context, this, ()| Ok(this.get()));
    }

    fn name() -> &'static str {
        "position"
    }
}

impl std::fmt::Display for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "Position({}, {})", self.0, self.1)
    }
}
