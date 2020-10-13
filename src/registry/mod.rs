use std::collections::HashMap;

enum VariableData {
    Int(i32),
    Uint(usize),
    String(String),
    Float(f64)
}

struct Registry<DataType> {
    debug_name: &'static str,
    data: DataType
}