use std::collections::HashMap;

use rlua::prelude::*;

#[derive(Debug, Deserialize, Clone)]
pub enum Value {
    Int(i64),
    OptionalInt(Option<i64>),
    Float(f64),
    OptionalFloat(Option<f64>),
    Str(String),
    OptionalStr(Option<String>),
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

impl Value {
    fn convert<'lua>(
        &self,
        value: LuaValue<'lua>,
        context: LuaContext<'lua>,
    ) -> Result<Value, LuaError> {
        match self {
            Value::Int(_) => Ok(Value::Int(
                context
                    .coerce_integer(value)
                    .and_then(|x| x.ok_or(make_type_error("Nil", "Int")))?,
            )),
            Value::OptionalInt(_) => Ok(Value::OptionalInt(context.coerce_integer(value)?)),
            Value::Float(_) => Ok(Value::Float(
                context
                    .coerce_number(value)
                    .and_then(|x| x.ok_or(make_type_error("Nil", "Float")))?,
            )),
            Value::OptionalFloat(_) => Ok(Value::OptionalFloat(context.coerce_number(value)?)),
            Value::Str(_) => Ok(Value::Str(
                context
                    .coerce_string(value)
                    .and_then(|x| x.ok_or(make_type_error("Nil", "String")))?
                    .to_str()?
                    .to_owned(),
            )),
            Value::OptionalStr(_) => Ok(Value::OptionalStr(
                context
                    .coerce_string(value)?
                    .map(|x| Ok(x.to_str()?.to_owned()))
                    .transpose()?,
            )),
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
                    panic!("__newindex method failed, terminating");
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
