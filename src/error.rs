#[derive(Debug)]
pub struct NoneError;

impl std::fmt::Display for NoneError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "No value found")
    }
}

impl std::error::Error for NoneError {}
