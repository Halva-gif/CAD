import FreeCAD as App
import Part


def make_glass_profile():
    # Coordinates are in the XY plane:
    # X is the radial direction; Y is the glass height.
    p0 = App.Vector(25, 0, 0)    # outer base
    p1 = App.Vector(38, 100, 0)  # outer rim
    p2 = App.Vector(35, 100, 0)  # inner rim
    p3 = App.Vector(4, 3, 0)     # inner base
    p4 = App.Vector(0, 3, 0)     # inner bottom at axis
    p5 = App.Vector(0, 0, 0)     # outer bottom at axis

    # Each arc ends at the same shared vertex where the next arc begins.
    outer_wall = Part.Arc(
        p0, App.Vector(31, 40, 0), p1
    )
    rim = Part.Arc(
        p1, App.Vector(36.5, 100.5, 0), p2
    )
    inner_wall = Part.Arc(
        p2, App.Vector(29, 45, 0), p3
    )
    inner_bottom = Part.Arc(
        p3, App.Vector(2, 2.5, 0), p4
    )
    bottom_inner_to_axis = Part.Arc(
        p4, App.Vector(0.1, 1.5, 0), p5
    )
    bottom_outer = Part.Arc(
        p5, App.Vector(12.5, -0.5, 0), p0
    )

    # The arcs form a closed cross-sectional shape.
    return Part.Shape([
        outer_wall,
        rim,
        inner_wall,
        inner_bottom,
        bottom_inner_to_axis,
        bottom_outer
    ])


glass_profile = make_glass_profile()
