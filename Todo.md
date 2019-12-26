# Renderer

# Scripting API
* Fill out lua APIs for existing systems
* Add event handlers in Lua
* Add timers, both native and lua
* Consideration for button presses: do we want to respond to them only once, or continually while held down?

# New Systems
* Add tile renderer for maps
* Add AI system
* Camera system
* Async resource loading

# Misc
* fix window selection problem

# Refactoring
* Consider abstracting away GameObjectComponent with a special storage type
* Fix GraphicsBuilder API so that it only returns a result on build()
* Clean up logging. Try to reduce the number of average fps and ignored window events
