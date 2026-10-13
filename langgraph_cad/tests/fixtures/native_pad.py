"""Test-only stand-in for an agent-written program; not a generation template."""
import FreeCAD as App
import Part

CAD_PARAMETERS = {
    'length': {'value': 20, 'unit': 'mm'},
    'width': {'value': 10, 'unit': 'mm'},
    'height': {'value': 5, 'unit': 'mm'},
}

doc = App.newDocument('NativePadTest')
body = doc.addObject('PartDesign::Body', 'Body')
sketch = body.newObject('Sketcher::SketchObject', 'BaseSketch')
length = CAD_PARAMETERS['length']['value']
width = CAD_PARAMETERS['width']['value']
height = CAD_PARAMETERS['height']['value']
points = [App.Vector(0, 0, 0), App.Vector(length, 0, 0),
          App.Vector(length, width, 0), App.Vector(0, width, 0)]
for i in range(4):
    sketch.addGeometry(Part.LineSegment(points[i], points[(i+1) % 4]), False)
pad = body.newObject('PartDesign::Pad', 'Pad')
pad.Profile = sketch
pad.Length = height
doc.recompute()
