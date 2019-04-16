use specs::prelude::*;

pub struct Presence;

impl Component for Presence {
    type Storage = VecStorage<Self>;
}

pub struct Space {
    passable: bool,
    contents: Option<Entity>,
}

impl Component for Space {
    type Storage = VecStorage<Self>;
}

impl Space {
    pub fn new(passable: bool) -> Space {
        Space {
            passable,
            contents: None,
        }
    }

    pub fn new_with_contents(passable: bool, contents: Entity) -> Space {
        Space {
            passable,
            contents: Some(contents),
        }
    }
}

pub struct Move {
    to: Entity,
}

impl Move {
    pub fn to(e: Entity) -> Move {
        Move { to: e }
    }
}

impl Component for Move {
    type Storage = VecStorage<Self>;
}

pub struct MapSystem;

impl<'a> System<'a> for MapSystem {
    type SystemData = (
        Entities<'a>,
        WriteStorage<'a, Space>,
        WriteStorage<'a, Move>,
    );

    fn run(&mut self, (entities, mut spaces, mut moves): Self::SystemData) {
        let mut to_remove = vec![];

        for (entity, mv) in (&entities, &moves).join() {
            trace!("Moving character...");
            let mut contents: Option<Entity> = None;

            if let Some(current) = spaces.get_mut(entity) {
                contents = current.contents;
                current.contents = None;
            } else {
                warn!("Failing to remove contents from space due to missing contents")
            }

            if let Some(target) = spaces.get_mut(mv.to) {
                if target.passable == true && target.contents.is_none() {
                    target.contents = contents;
                    trace!("Character moved!");
                } else {
                    warn!("Cannot move to target space!");
                }
            } else {
                warn!("Failed to find target space!");
            }

            to_remove.push(entity);
        }

        for mv in to_remove {
            moves.remove(mv);
        }
    }
}
