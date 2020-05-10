# Renderer
* Separate out 2D rendering pipeline into it's own module
* Replace the recreate_swapchain flag with an event triggered by the window and sent to the renderer
* Create a 3D rendering module

# Scripting API
* Fill out lua APIs for existing systems
* Add timers, both native and lua
* Consideration for button presses: do we want to respond to them only once, or continually while held down?
* Merge native and lua scripting APIs
* Look into possible deadlock issues with big, complicated scenes (replace Mutex<T> with RwLock<T> ?)

# New Systems
* Add tile renderer for maps
* Add AI system
* Camera system
* Async resource loading (tokio, but is this necessary?)

# Misc

# Refactoring
* Consider abstracting away GameObjectComponent with a special storage type
* Clean up logging. Try to reduce the number of average fps and ignored window events
* See what else can be tested.
