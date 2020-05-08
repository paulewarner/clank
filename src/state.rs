use std::collections::HashMap;

use rlua::prelude::*;

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    OptionalInt(Option<i64>),
    Float(f64),
    OptionalFloat(Option<f64>),
    Str(String),
    OptionalStr(Option<String>),
    Bool(bool),
    OptionalBool(Option<bool>),
}

impl<'lua> ToLua<'lua> for Value {
    fn to_lua(self, context: LuaContext<'lua>) -> Result<LuaValue<'lua>, LuaError> {
        Ok(match self {
            Value::Int(i) => LuaValue::Integer(i),
            Value::OptionalInt(i) => i.map(|x| LuaValue::Integer(x)).unwrap_or(LuaValue::Nil),
            Value::Float(i) => LuaValue::Number(i),
            Value::OptionalFloat(i) => i.map(|x| LuaValue::Number(x)).unwrap_or(LuaValue::Nil),
            Value::Str(i) => LuaValue::String(context.create_string(&i)?),
            Value::OptionalStr(i) => i
                .map(|x| context.create_string(&x).map(|y| LuaValue::String(y)))
                .unwrap_or(Ok(LuaValue::Nil))?,
            Value::Bool(b) => LuaValue::Boolean(b),
            Value::OptionalBool(i) => i.map(|x| LuaValue::Boolean(x)).unwrap_or(LuaValue::Nil),
        })
    }
}

fn make_type_error(from: &'static str, to: &'static str) -> LuaError {
    LuaError::FromLuaConversionError {
        from,
        to,
        message: None,
    }
}

fn get_int<'lua>(context: LuaContext<'lua>, v: LuaValue<'lua>) -> Result<Option<i64>, LuaError> {
    match v {
        LuaValue::Nil => Ok(None),
        _ => Ok(Some(
            context
                .coerce_integer(v)
                .and_then(|x| x.ok_or(make_type_error("Nil", "Int")))?,
        )), // Must always have value here.
    }
}

fn get_float<'lua>(context: LuaContext<'lua>, v: LuaValue<'lua>) -> Result<Option<f64>, LuaError> {
    match v {
        LuaValue::Nil => Ok(None),
        _ => Ok(Some(
            context
                .coerce_number(v)
                .and_then(|x| x.ok_or(make_type_error("Nil", "Float")))?,
        )), // Must always have value here.
    }
}

fn get_str<'lua>(v: LuaValue<'lua>) -> Result<Option<LuaString<'lua>>, LuaError> {
    match v {
        LuaValue::Nil => Ok(None),
        LuaValue::String(s) => Ok(Some(s)),
        _ => Err(make_type_error("Nil", "String")),
    }
}

impl Value {
    fn convert<'lua>(
        &self,
        value: LuaValue<'lua>,
        context: LuaContext<'lua>,
    ) -> Result<Value, LuaError> {
        match self {
            Value::Int(_) => Ok(Value::Int(
                get_int(context, value).and_then(|x| x.ok_or(make_type_error("Nil", "Int")))?,
            )),
            Value::OptionalInt(_) => Ok(Value::OptionalInt(get_int(context, value)?)),
            Value::Float(_) => Ok(Value::Float(
                get_float(context, value).and_then(|x| x.ok_or(make_type_error("Nil", "Float")))?,
            )),
            Value::OptionalFloat(_) => Ok(Value::OptionalFloat(get_float(context, value)?)),
            Value::Str(_) => Ok(Value::Str(
                get_str(value)
                    .and_then(|x| x.ok_or(make_type_error("Nil", "String")))?
                    .to_str()?
                    .to_owned(),
            )),
            Value::OptionalStr(_) => Ok(Value::OptionalStr(
                get_str(value)?
                    .map(|x| Ok(x.to_str()?.to_owned()))
                    .transpose()?,
            )),
            Value::Bool(_) => match value {
                LuaValue::Boolean(b) => Ok(Value::Bool(b)),
                _ => Err(make_type_error("other", "bool")),
            },
            Value::OptionalBool(_) => match value {
                LuaValue::Boolean(b) => Ok(Value::OptionalBool(Some(b))),
                LuaValue::Nil => Ok(Value::OptionalBool(None)),
                _ => Err(make_type_error("other", "Optional Bool")),
            },
        }
    }
}

// This is a workaround for serde deserialization. Basically, we need to specify where to get the information, which we can then convert into the proper type later
#[derive(Debug, Deserialize)]
pub enum FieldDef {
    Int {
        name: String,
        #[serde(rename = "$value")]
        v: i64,
    },
    OptionalInt {
        name: String,
        #[serde(rename = "$value")]
        v: Option<i64>,
    },
    Float {
        name: String,
        #[serde(rename = "$value")]
        v: f64,
    },
    OptionalFloat {
        name: String,
        #[serde(rename = "$value")]
        v: Option<f64>,
    },
    Str {
        name: String,
        #[serde(rename = "$value")]
        v: String,
    },
    OptionalStr {
        name: String,
        #[serde(rename = "$value")]
        v: Option<String>,
    },
}

#[derive(Debug, Deserialize)]
#[serde(from = "FieldDef")]
pub struct Field {
    pub name: String,
    pub value: Value,
}

impl Field {
    fn new<S: AsRef<str>>(name: S, value: Value) -> Field {
        Field {
            name: String::from(name.as_ref()),
            value,
        }
    }
}

impl std::convert::From<FieldDef> for Field {
    fn from(def: FieldDef) -> Self {
        match def {
            FieldDef::Int { name, v } => Field {
                name: name,
                value: Value::Int(v),
            },
            FieldDef::OptionalInt { name, v } => Field {
                name: name,
                value: Value::OptionalInt(v),
            },
            FieldDef::Float { name, v } => Field {
                name: name,
                value: Value::Float(v),
            },
            FieldDef::OptionalFloat { name, v } => Field {
                name: name,
                value: Value::OptionalFloat(v),
            },
            FieldDef::Str { name, v } => Field {
                name: name,
                value: Value::Str(v),
            },
            FieldDef::OptionalStr { name, v } => Field {
                name: name,
                value: Value::OptionalStr(v),
            },
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ScriptFields {
    #[serde(rename = "$value")]
    pub fields: Vec<Field>,
}

impl ScriptFields {
    pub fn new(fields: Vec<Field>) -> ScriptFields {
        ScriptFields { fields }
    }
}

impl Into<std::collections::HashMap<String, Value>> for ScriptFields {
    fn into(self) -> std::collections::HashMap<String, Value> {
        self.fields.into_iter().map(|x| (x.name, x.value)).collect()
    }
}

#[derive(Clone)]
pub struct ScriptState {
    state: std::sync::Arc<std::sync::Mutex<HashMap<String, Value>>>,
}

impl LuaUserData for ScriptState {}

impl ScriptState {
    pub fn new(fields: ScriptFields) -> ScriptState {
        ScriptState {
            state: std::sync::Arc::new(std::sync::Mutex::new(fields.into())),
        }
    }

    fn metatable<'lua>(context: &LuaContext<'lua>) -> Result<LuaTable<'lua>, LuaError> {
        let table = context.create_table()?;
        table.set(
            "__index",
            context.create_function(|context, args| match index_private(context, args) {
                Ok(value) => Ok(value),
                Err(e) => {
                    error!("Error occured in __index method: {:?}", e);
                    panic!("__index method failed, terminating");
                }
            })?,
        )?;

        table.set(
            "__newindex",
            context.create_function(|context, args| match newindex_private(context, args) {
                Ok(value) => Ok(value),
                Err(e) => {
                    error!("Error occured in __newindex method: {:?}", e);
                    Err(e)
                    // panic!("__newindex method failed, terminating");
                }
            })?,
        )?;
        Ok(table)
    }

    pub fn to_table<'lua>(self, context: &LuaContext<'lua>) -> Result<LuaTable<'lua>, LuaError> {
        let table = context.create_table()?;
        table.set("__inner", self)?;
        table.set_metatable(Some(ScriptState::metatable(context)?));
        Ok(table)
    }
}

fn index_private<'lua>(
    context: LuaContext<'lua>,
    (table, key): (LuaTable<'lua>, LuaString<'lua>),
) -> Result<LuaValue<'lua>, LuaError> {
    let userdata: LuaAnyUserData<'lua> = table.raw_get("__inner")?;
    let state = userdata.borrow::<ScriptState>()?;
    let map = state.state.lock().expect("Failed to lock mutex");
    match map.get(key.to_str()?) {
        // TODO: I can't get around this clone here, can I...
        Some(value) => Ok(value.clone().to_lua(context)?),
        None => Ok(LuaNil),
    }
}

fn newindex_private<'lua>(
    context: LuaContext<'lua>,
    (table, key, new_value): (LuaTable<'lua>, LuaString<'lua>, LuaValue<'lua>),
) -> Result<LuaValue<'lua>, LuaError> {
    let userdata: LuaAnyUserData<'lua> = table.raw_get("__inner")?;
    let state = userdata.borrow_mut::<ScriptState>()?;
    let s = key.to_str()?;
    let mut map = state.state.lock().expect("Failed to lock mutex");
    let old_value = map.get_mut(s).expect("No such field!");
    match old_value.convert(new_value, context) {
        Ok(value) => {
            map.insert(s.to_owned(), value);
            Ok(LuaNil)
        }
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lua_test<F: for<'lua> FnOnce(LuaContext<'lua>)>(f: F) {
        let lua = Lua::new();
        lua.context(f);
    }

    fn test_script_state<
        C: AsRef<[u8]>,
        F: for<'lua> FnOnce(LuaContext<'lua>, ScriptState, LuaValue<'lua>),
    >(
        fields: Vec<Field>,
        chunk: C,
        f: F,
        should_fail: bool,
    ) {
        lua_test(|context| {
            println!("{}", std::str::from_utf8(chunk.as_ref()).unwrap());
            let state = ScriptState::new(ScriptFields::new(fields));
            let table = state.clone().to_table(&context).unwrap();
            let chunk: LuaFunction = context.load(chunk.as_ref()).eval().unwrap();
            match chunk.call(table) {
                Ok(r) => {
                    if should_fail {
                        panic!("Unexpected success: {:?}", state.state.lock());
                    }
                    f(context, state, r);
                }
                Err(e) => {
                    if !should_fail {
                        panic!("Function test failed with error {:?}", e);
                    }
                }
            };
        });
    }

    fn test_assignment<S: AsRef<str>>(
        starting_value: Value,
        ending_value: Value,
        assignment: S,
        should_fail: bool,
    ) {
        test_script_state(
            vec![Field::new("a", starting_value)],
            format!(
                r#"
            function(test)
                test.a = {}
            end
            "#,
                assignment.as_ref()
            ),
            |_context, state, ret| {
                match ret {
                    LuaValue::Nil => (),
                    _ => panic!("Expected set to return nil"),
                };
                let s = state.state.lock().unwrap();
                assert_eq!(s.get("a").unwrap(), &ending_value)
            },
            should_fail,
        );
    }

    #[test]
    fn test_tolua_int() {
        lua_test(|context| {
            let i = 1;
            match Value::Int(i).to_lua(context).unwrap() {
                LuaValue::Integer(n) => assert_eq!(n, i),
                _ => panic!("Integer {} converted to invalid type", i),
            };
        });
    }

    #[test]
    fn test_tolua_optionalint() {
        lua_test(|context| {
            let i = 1;
            match Value::OptionalInt(Some(i)).to_lua(context).unwrap() {
                LuaValue::Integer(n) => assert_eq!(n, i),
                _ => panic!("Optional Int {} converted to invalid type", i),
            };
            match Value::OptionalInt(None).to_lua(context).unwrap() {
                LuaValue::Nil => (),
                _ => panic!("Optional Int (nil) converted to invalid type"),
            };
        });
    }

    #[test]
    fn test_tolua_float() {
        lua_test(|context| {
            let i = 1.0;
            match Value::Float(i).to_lua(context).unwrap() {
                LuaValue::Number(n) => assert_eq!(n, i),
                _ => panic!("Float {} converted to invalid type", i),
            };
        });
    }

    #[test]
    fn test_tolua_optionalfloat() {
        lua_test(|context| {
            let i = 1.0;
            match Value::OptionalFloat(Some(i)).to_lua(context).unwrap() {
                LuaValue::Number(n) => assert_eq!(n, i),
                _ => panic!("Optional Int {} converted to invalid type", i),
            };
            match Value::OptionalFloat(None).to_lua(context).unwrap() {
                LuaValue::Nil => (),
                _ => panic!("Optional Int (nil) converted to invalid type"),
            };
        });
    }

    #[test]
    fn test_tolua_str() {
        lua_test(|context| {
            let i = String::from("string");
            match Value::Str(i.clone()).to_lua(context).unwrap() {
                LuaValue::String(n) => assert_eq!(n, i),
                _ => panic!("String {} converted to invalid type", i),
            };
        });
    }

    #[test]
    fn test_tolua_optionalstr() {
        lua_test(|context| {
            let i = String::from("string");
            match Value::OptionalStr(Some(i.clone())).to_lua(context).unwrap() {
                LuaValue::String(n) => assert_eq!(n, i),
                _ => panic!("Optional String {} converted to invalid type", i),
            };
            match Value::OptionalStr(None).to_lua(context).unwrap() {
                LuaValue::Nil => (),
                _ => panic!("Optional String (nil) converted to invalid type"),
            };
        });
    }

    #[test]
    fn test_index_set_int() {
        test_assignment(Value::Int(0), Value::Int(1), "1", false);
    }

    #[test]
    fn test_index_set_optional_none_to_int() {
        test_assignment(
            Value::OptionalInt(None),
            Value::OptionalInt(Some(1)),
            "1",
            false,
        );
    }

    #[test]
    fn test_index_set_optional_int_to_none() {
        test_assignment(
            Value::OptionalInt(Some(1)),
            Value::OptionalInt(None),
            "nil",
            false,
        );
    }

    #[test]
    fn test_index_set_float() {
        test_assignment(Value::Float(1.5), Value::Float(3.5), "3.5", false);
    }

    #[test]
    fn test_index_set_optional_none_to_float() {
        test_assignment(
            Value::OptionalFloat(None),
            Value::OptionalFloat(Some(3.5)),
            "3.5",
            false,
        );
    }

    #[test]
    fn test_index_set_optional_float_to_none() {
        test_assignment(
            Value::OptionalFloat(Some(3.5)),
            Value::OptionalFloat(None),
            "nil",
            false,
        );
    }

    #[test]
    fn test_index_set_str() {
        test_assignment(
            Value::Str(String::from("a")),
            Value::Str(String::from("b")),
            "\"b\"",
            false,
        );
    }

    #[test]
    fn test_index_set_optional_none_to_str() {
        test_assignment(
            Value::OptionalStr(None),
            Value::OptionalStr(Some(String::from("a"))),
            "\"a\"",
            false,
        );
    }

    #[test]
    fn test_index_set_optional_str_to_none() {
        test_assignment(
            Value::OptionalStr(Some(String::from("a"))),
            Value::OptionalStr(None),
            "nil",
            false,
        );
    }

    #[test]
    fn test_index_set_bool() {
        test_assignment(Value::Bool(true), Value::Bool(false), "false", false);
    }

    #[test]
    fn test_index_set_optional_none_to_bool() {
        test_assignment(
            Value::OptionalBool(None),
            Value::OptionalBool(Some(true)),
            "true",
            false,
        );
    }

    #[test]
    fn test_index_set_optional_bool_to_none() {
        test_assignment(
            Value::OptionalBool(Some(true)),
            Value::OptionalBool(None),
            "nil",
            false,
        );
    }

    #[test]
    fn test_index_set_int_fail() {
        test_assignment(Value::Int(1), Value::OptionalInt(None), "nil", true);
    }

    #[test]
    fn test_index_set_optional_int_fail() {
        test_assignment(
            Value::OptionalInt(None),
            Value::OptionalStr(Some(String::from("a"))),
            "\"a\"",
            true,
        );
    }

    #[test]
    fn test_index_set_float_fail() {
        test_assignment(Value::Float(1.0), Value::OptionalFloat(None), "nil", true);
    }

    #[test]
    fn test_index_set_optional_float_fail() {
        test_assignment(
            Value::OptionalFloat(None),
            Value::OptionalStr(Some(String::from("a"))),
            "\"a\"",
            true,
        );
    }

    #[test]
    fn test_index_set_str_fail() {
        test_assignment(
            Value::Str(String::from("a")),
            Value::OptionalStr(None),
            "nil",
            true,
        );
    }

    #[test]
    fn test_index_set_optional_str_fail() {
        test_assignment(
            Value::OptionalStr(None),
            Value::OptionalFloat(Some(1.0)),
            "1.0",
            true,
        );
    }
}
