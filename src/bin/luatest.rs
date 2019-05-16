use std::path::Path;
use std::io::prelude::*;
use std::io;
use std::fs::File;
use std::io::BufReader;
use std::sync::{Arc, Mutex};

use rlua::prelude::*;

fn load_file<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
    let mut vec = Vec::new();
    BufReader::new(File::open(path)?).read_to_end(&mut vec)?;
    Ok(vec)
}

trait Scriptable: Sized {
    fn add_clanks<'a, 'lua, M: LuaUserDataMethods<'lua, LuaStruct<Self>>>(methods: &'a mut MethodAdder<'a, 'lua, Self, M>);
}

struct _MethodAdder<'a, 'lua, F: 'lua, T: Scriptable, M> where M: LuaUserDataMethods<'lua, LuaStruct<T>> {
    methods: &'a mut M,
    phantom: std::marker::PhantomData<T>,
    phanto: std::marker::PhantomData<&'lua F>,
}

type MethodAdder<'a, 'lua, T, M> = _MethodAdder<'a, 'lua, (), T, M>;

impl<'a, 'lua, T: Scriptable, M: LuaUserDataMethods<'lua, LuaStruct<T>>> MethodAdder<'a, 'lua, T, M> {
    fn add_method<S: ?Sized, A, R, F>(&mut self, name: &S, method: F) where
        S: AsRef<[u8]>,
        A: FromLuaMulti<'lua>,
        R: ToLuaMulti<'lua>,
        F: 'static + Send + Sync + Fn(LuaContext<'lua>, &T, A) -> LuaResult<R> {
        self.methods.add_method(name, move |context, wrapper, arg| {
            let item = wrapper.item.lock().expect("Failed to lock mutex");
            method(context, &item, arg)
        })
    }

    fn add_method_mut<S: ?Sized, A, R, F>(&mut self, name: &S, method: F) where
        S: AsRef<[u8]>,
        A: FromLuaMulti<'lua>,
        R: ToLuaMulti<'lua>,
        F: 'static + Send + Sync + Fn(LuaContext<'lua>, &mut T, A) -> LuaResult<R> {
        self.methods.add_method_mut(name, move |context, wrapper, arg| {
            let mut item = wrapper.item.lock().expect("Failed to lock mutex");
            method(context, &mut item, arg)
        })
    }

    fn add_function<S: ?Sized, A, R, F>(&mut self, name: &S, function: F) where
        S: AsRef<[u8]>,
        A: FromLuaMulti<'lua>,
        R: ToLuaMulti<'lua>,
        F: 'static + Send + Fn(LuaContext<'lua>, A) -> LuaResult<R> {
        self.methods.add_function(name, function)
    }

    fn add_function_mut<S: ?Sized, A, R, F>(&mut self, name: &S, mut function: F) where
        S: AsRef<[u8]>,
        A: FromLuaMulti<'lua>,
        R: ToLuaMulti<'lua>,
        F: 'static + Send + FnMut(LuaContext<'lua>, A) -> LuaResult<R> {
        self.methods.add_function_mut(name, function)
    }

    fn add_meta_method<A, R, F>(&mut self, meta: LuaMetaMethod, method: F) where
        A: FromLuaMulti<'lua>,
        R: ToLuaMulti<'lua>,
        F: 'static + Send + Sync + Fn(LuaContext<'lua>, &T, A) -> LuaResult<R> {
            self.methods.add_meta_method(meta, move |context, wrapper, arg| {
                let item = wrapper.item.lock().expect("Failed to lock mutex");
                method(context, &item, arg)
            })
    }

    fn add_meta_method_mut<A, R, F>(&mut self, meta: LuaMetaMethod, method: F) where
        A: FromLuaMulti<'lua>,
        R: ToLuaMulti<'lua>,
        F: 'static + Send + Sync + Fn(LuaContext<'lua>, &mut T, A) -> LuaResult<R> {
            self.methods.add_meta_method_mut(meta, move |context, wrapper, arg| {
                let mut item = wrapper.item.lock().expect("Failed to lock mutex");
                method(context, &mut item, arg)
            })
    }

    fn add_meta_function<A, R, F>(&mut self, meta: LuaMetaMethod, function: F) where
        A: FromLuaMulti<'lua>,
        R: ToLuaMulti<'lua>,
        F: 'static + Send + Fn(LuaContext<'lua>, A) -> LuaResult<R> {
        self.methods.add_meta_function(meta, function)
    }

    fn add_meta_function_mut<A, R, F>(&mut self, meta: LuaMetaMethod, function: F) where
        A: FromLuaMulti<'lua>,
        R: ToLuaMulti<'lua>,
        F: 'static + Send + FnMut(LuaContext<'lua>, A) -> LuaResult<R> {
        self.methods.add_meta_function_mut(meta, function)
    }
}

#[derive(Debug, Clone)]
struct LuaStruct<T: Scriptable> {
    item: Arc<Mutex<T>>,
}

impl<T: Scriptable> LuaUserData for LuaStruct<T> {
    fn add_methods<'lua, M: LuaUserDataMethods<'lua, Self>>(methods: &mut M) {
        let mut method_adder = MethodAdder {
            methods,
            phantom: std::marker::PhantomData,
            phanto: std::marker::PhantomData
        };

        T::add_clanks(&mut method_adder)
    }
}

impl<T: Scriptable> LuaStruct<T> {
    fn new(item: Arc<Mutex<T>>) -> LuaStruct<T> {
        LuaStruct {
            item: item
        }
    }
}

#[derive(Debug)]
struct Position {
    x: u32,
    y: u32
}

impl LuaUserData for Position {}

impl Scriptable for Position {
    fn add_clanks<'a, 'lua, M: LuaUserDataMethods<'lua, LuaStruct<Self>>>(methods: &'a mut MethodAdder<'a, 'lua, Self, M>) {
        methods.add_method("get_x", |_context, this, _: ()| {
            Ok(this.x)
        });

        methods.add_method_mut("set_x", |_context, this, x: u32| {
            this.x = x;
            Ok(())
        });

        methods.add_method("get_y", |_, this, _: ()| {
            Ok(this.y)
        });
    }
}

fn main() {
    let lua = Lua::new();
    lua.context(|context| {
        let script = load_file("scripts/test.lua").expect("Failed to load script");
        let chunk = context.load(&script);
        let globals = context.globals();
        let point = Arc::new(Mutex::new(Position {
            x: 0,
            y: 0,
        }));

        let m = LuaStruct::new(point.clone());
        globals.set("position", m).expect("Failed to set global variable");
        chunk.exec().expect("Failed to execute chunk");
        println!("{:?}", point.lock().unwrap());
    });
}