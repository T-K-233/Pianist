#!/usr/bin/env python3
"""
Advanced MJCF to URDF converter specifically designed for piano models.
This script converts the piano MJCF file to URDF format with proper joint hierarchy,
materials, and collision properties.
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


class MjcfToUrdfConverter:
    """Converts MJCF files to URDF format with proper handling of piano-specific elements."""

    def __init__(self):
        self.materials = {}
        self.defaults = {}
        self.bodies = []
        self.joints = []

    def parse_mjcf(self, mjcf_path: str) -> ET.Element:
        """Parse the MJCF XML file."""
        tree = ET.parse(mjcf_path)
        root = tree.getroot()

        # Extract materials
        self._extract_materials(root)

        # Extract default classes
        self._extract_defaults(root)

        return root

    def _extract_materials(self, root: ET.Element):
        """Extract material definitions from the MJCF file."""
        asset = root.find("asset")
        if asset is not None:
            for material in asset.findall("material"):
                name = material.get("name")
                rgba = material.get("rgba", "0.5 0.5 0.5 1")
                self.materials[name] = rgba

    def _extract_defaults(self, root: ET.Element):
        """Extract default class definitions from the MJCF file."""
        default_elem = root.find("default")
        if default_elem is not None:
            # Process nested defaults recursively
            self._process_default_class(default_elem, "/")

    def _process_default_class(self, default_class: ET.Element, parent_class: str):
        """Process a default class and its nested classes."""
        class_name = default_class.get("class", parent_class)
        self.defaults[class_name] = {}

        # Extract joint properties
        joint = default_class.find("joint")
        if joint is not None:
            self.defaults[class_name]["joint"] = {
                "type": joint.get("type", "revolute"),
                "axis": joint.get("axis", "0 1 0"),
                "pos": joint.get("pos", "0 0 0"),
                "range": joint.get("range", "0 1"),
                "damping": joint.get("damping", "0"),
                "armature": joint.get("armature", "0"),
                "stiffness": joint.get("stiffness", "0"),
                "springref": joint.get("springref", "0")
            }

        # Extract geom properties
        geom = default_class.find("geom")
        if geom is not None:
            self.defaults[class_name]["geom"] = {
                "type": geom.get("type", "box"),
                "size": geom.get("size", "0.1 0.1 0.1"),
                "material": geom.get("material", ""),
                "mass": geom.get("mass", "1.0")
            }

        # Process nested default classes
        for nested_default in default_class.findall("default"):
            nested_class_name = nested_default.get("class")
            if nested_class_name:
                self._process_default_class(nested_default, class_name)

    def _get_body_class(self, body: ET.Element) -> str:
        """Determine the default class for a body based on its geometry."""
        geom = body.find("geom")
        if geom is not None:
            geom_class = geom.get("class", "/")
            if geom_class in self.defaults:
                return geom_class
        return "/"

    def _create_link(self, robot: ET.Element, body: ET.Element, body_name: str, body_class: str):
        """Create a URDF link from an MJCF body."""
        link = ET.SubElement(robot, "link")
        link.set("name", body_name)

        # Get geometry properties from the body's geom element or defaults
        geom_element = body.find("geom")
        if geom_element is not None:
            # Check if geom has explicit attributes, otherwise use class defaults
            geom_type = geom_element.get("type")
            geom_size = geom_element.get("size")
            geom_material = geom_element.get("material")

            # If not explicitly set, use class defaults
            if geom_type is None and body_class in self.defaults and "geom" in self.defaults[body_class]:
                geom_type = self.defaults[body_class]["geom"]["type"]
            if geom_size is None and body_class in self.defaults and "geom" in self.defaults[body_class]:
                geom_size = self.defaults[body_class]["geom"]["size"]
            if geom_material is None and body_class in self.defaults and "geom" in self.defaults[body_class]:
                geom_material = self.defaults[body_class]["geom"]["material"]

            # Set defaults if still None
            if geom_type is None:
                geom_type = "box"
            if geom_size is None:
                geom_size = "0.1 0.1 0.1"
            if geom_material is None:
                geom_material = ""
        elif body_class in self.defaults and "geom" in self.defaults[body_class]:
            geom_props = self.defaults[body_class]["geom"]
            geom_type = geom_props["type"]
            geom_size = geom_props["size"]
            geom_material = geom_props["material"]
        else:
            geom_type = "box"
            geom_size = "0.1 0.1 0.1"
            geom_material = ""

        # Add visual
        visual = ET.SubElement(link, "visual")
        geometry = ET.SubElement(visual, "geometry")

        if geom_type == "box":
            box = ET.SubElement(geometry, "box")
            # MJCF uses half-dimensions, URDF uses full dimensions
            size_parts = geom_size.split()
            if len(size_parts) >= 3:
                full_size = f"{float(size_parts[0])*2:.6f} {float(size_parts[1])*2:.6f} {float(size_parts[2])*2:.6f}"
                box.set("size", full_size)
            else:
                box.set("size", geom_size)

        if geom_material and geom_material in self.materials:
            material = ET.SubElement(visual, "material")
            material.set("name", geom_material)

        # Add collision
        collision = ET.SubElement(link, "collision")
        collision_geometry = ET.SubElement(collision, "geometry")

        if geom_type == "box":
            collision_box = ET.SubElement(collision_geometry, "box")
            # MJCF uses half-dimensions, URDF uses full dimensions
            size_parts = geom_size.split()
            if len(size_parts) >= 3:
                full_size = f"{float(size_parts[0])*2:.6f} {float(size_parts[1])*2:.6f} {float(size_parts[2])*2:.6f}"
                collision_box.set("size", full_size)
            else:
                collision_box.set("size", geom_size)

        # Add inertial properties
        inertial = ET.SubElement(link, "inertial")
        mass = ET.SubElement(inertial, "mass")

        # Get mass from geom element or defaults
        if geom_element is not None:
            mass_value = geom_element.get("mass")
            if mass_value is None and body_class in self.defaults and "geom" in self.defaults[body_class]:
                mass_value = self.defaults[body_class]["geom"]["mass"]
            if mass_value is None:
                mass_value = "1.0"
        elif body_class in self.defaults and "geom" in self.defaults[body_class]:
            mass_value = self.defaults[body_class]["geom"]["mass"]
        else:
            mass_value = "1.0"

        mass.set("value", mass_value)

        # Calculate inertia for box geometry using the full dimensions
        geom_size_parts = geom_size.split()
        if len(geom_size_parts) >= 3:
            # Convert half-dimensions to full dimensions for inertia calculation
            lx, ly, lz = float(geom_size_parts[0])*2, float(geom_size_parts[1])*2, float(geom_size_parts[2])*2
            mass_val = float(mass_value)

            # Box inertia formulas (using full dimensions)
            ixx = mass_val * (ly**2 + lz**2) / 12.0
            iyy = mass_val * (lx**2 + lz**2) / 12.0
            izz = mass_val * (lx**2 + ly**2) / 12.0

            inertia = ET.SubElement(inertial, "inertia")
            inertia.set("ixx", f"{ixx:.6f}")
            inertia.set("ixy", "0.0")
            inertia.set("ixz", "0.0")
            inertia.set("iyy", f"{iyy:.6f}")
            inertia.set("iyz", "0.0")
            inertia.set("izz", f"{izz:.6f}")
        else:
            # Default inertia
            inertia = ET.SubElement(inertial, "inertia")
            inertia.set("ixx", "0.001")
            inertia.set("ixy", "0.0")
            inertia.set("ixz", "0.0")
            inertia.set("iyy", "0.001")
            inertia.set("iyz", "0.0")
            inertia.set("izz", "0.001")

        return link

    def _create_joint(self, robot: ET.Element, body: ET.Element, body_name: str, body_class: str):
        """Create a URDF joint from an MJCF body."""
        joint = ET.SubElement(robot, "joint")
        joint.set("name", f"{body_name}_joint")
        joint.set("type", "revolute")

        parent = ET.SubElement(joint, "parent")
        parent.set("link", "base")

        child = ET.SubElement(joint, "child")
        child.set("link", body_name)

        # Calculate origin position
        # In MJCF, all bodies are independent, so we use the body position directly
        body_pos = body.get("pos", "0 0 0").split()
        body_pos = [float(x) for x in body_pos]

        # The joint origin should be the body position (since all bodies are independent in MJCF)
        origin_pos = body_pos

        origin = ET.SubElement(joint, "origin")
        origin.set("xyz", f"{origin_pos[0]} {origin_pos[1]} {origin_pos[2]}")
        origin.set("rpy", "0 0 0")

        axis = ET.SubElement(joint, "axis")
        if body_class in self.defaults and "joint" in self.defaults[body_class]:
            joint_axis = self.defaults[body_class]["joint"]["axis"]
            axis.set("xyz", joint_axis)
        else:
            axis.set("xyz", "0 1 0")

        limit = ET.SubElement(joint, "limit")
        if body_class in self.defaults and "joint" in self.defaults[body_class]:
            joint_range = self.defaults[body_class]["joint"]["range"].split()
            limit.set("lower", joint_range[0])
            limit.set("upper", joint_range[1])
        else:
            limit.set("lower", "0")
            limit.set("upper", "1")

        # Set effort and velocity limits based on joint properties
        if body_class in self.defaults and "joint" in self.defaults[body_class]:
            stiffness = float(self.defaults[body_class]["joint"]["stiffness"])
            damping = float(self.defaults[body_class]["joint"]["damping"])

            # Calculate effort based on stiffness
            effort = max(10.0, stiffness * 0.1)
            velocity = max(1.0, 1.0 / (damping + 0.1))

            limit.set("effort", f"{effort:.2f}")
            limit.set("velocity", f"{velocity:.2f}")
        else:
            limit.set("effort", "10")
            limit.set("velocity", "1")

        return joint

    def convert_to_urdf(self, mjcf_path: str, output_path: str = None) -> str:
        """
        Convert MJCF file to URDF format.

        Args:
            mjcf_path: Path to the MJCF file
            output_path: Path for the output URDF file (optional)

        Returns:
            Path to the generated URDF file
        """
        if output_path is None:
            mjcf_file = Path(mjcf_path)
            output_path = mjcf_file.parent / f"{mjcf_file.stem}.urdf"

        # Parse MJCF
        root = self.parse_mjcf(mjcf_path)

        # Create URDF root element
        robot = ET.Element("robot")
        robot.set("name", root.get("model", "piano"))

        # Add materials to URDF
        for name, rgba in self.materials.items():
            material_elem = ET.SubElement(robot, "material")
            material_elem.set("name", name)
            color_elem = ET.SubElement(material_elem, "color")
            color_elem.set("rgba", rgba)

        # Process worldbody
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("No worldbody found in MJCF")

        # Add base link
        base_link = ET.SubElement(robot, "link")
        base_link.set("name", "base")

        # Add base link visual (from base body)
        base_body = worldbody.find("body[@name='base']")
        if base_body is not None:
            base_pos = base_body.get("pos", "0 0 0").split()
            base_pos = [float(x) for x in base_pos]

            base_geom = base_body.find("geom")
            if base_geom is not None:
                base_size = base_geom.get("size", "0.1 0.1 0.1")
                # Convert half-dimensions to full dimensions
                base_size_parts = base_size.split()
                if len(base_size_parts) >= 3:
                    full_base_size = f"{float(base_size_parts[0])*2:.6f} {float(base_size_parts[1])*2:.6f} {float(base_size_parts[2])*2:.6f}"
                else:
                    full_base_size = base_size

                # Add visual to base link with origin
                visual = ET.SubElement(base_link, "visual")
                origin = ET.SubElement(visual, "origin")
                origin.set("xyz", f"{base_pos[0]} {base_pos[1]} {base_pos[2]}")
                origin.set("rpy", "0 0 0")
                geometry = ET.SubElement(visual, "geometry")
                box = ET.SubElement(geometry, "box")
                box.set("size", full_base_size)

                # Add collision to base link with origin
                collision = ET.SubElement(base_link, "collision")
                collision_origin = ET.SubElement(collision, "origin")
                collision_origin.set("xyz", f"{base_pos[0]} {base_pos[1]} {base_pos[2]}")
                collision_origin.set("rpy", "0 0 0")
                collision_geometry = ET.SubElement(collision, "geometry")
                collision_box = ET.SubElement(collision_geometry, "box")
                collision_box.set("size", full_base_size)

                # Add inertial to base link with origin
                inertial = ET.SubElement(base_link, "inertial")
                inertial_origin = ET.SubElement(inertial, "origin")
                inertial_origin.set("xyz", f"{base_pos[0]} {base_pos[1]} {base_pos[2]}")
                inertial_origin.set("rpy", "0 0 0")
                mass = ET.SubElement(inertial, "mass")
                mass.set("value", "10.0")  # Base is heavier
                inertia = ET.SubElement(inertial, "inertia")
                inertia.set("ixx", "0.1")
                inertia.set("ixy", "0.0")
                inertia.set("ixz", "0.0")
                inertia.set("iyy", "0.1")
                inertia.set("iyz", "0.0")
                inertia.set("izz", "0.1")

        # Process all key bodies (skip base)
        for body in worldbody.findall("body"):
            body_name = body.get("name")
            if body_name == "base":
                continue

            body_class = self._get_body_class(body)

            # Create link
            link = self._create_link(robot, body, body_name, body_class)

            # Create joint
            joint = self._create_joint(robot, body, body_name, body_class)

            self.bodies.append(body_name)

        # Write URDF to file
        tree = ET.ElementTree(robot)
        ET.indent(tree, space="  ", level=0)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

        print(f"URDF file saved to: {output_path}")
        print(f"Converted {len(self.bodies)} piano keys to URDF format")
        print(f"Materials: {list(self.materials.keys())}")
        print(f"Default classes: {list(self.defaults.keys())}")

        return str(output_path)


def main():
    """Main function to run the converter."""
    parser = argparse.ArgumentParser(
        description="Convert piano MJCF file to URDF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_piano_mjcf_to_urdf.py piano_no_actuators.xml
  python convert_piano_mjcf_to_urdf.py piano_no_actuators.xml -o piano.urdf
  python convert_piano_mjcf_to_urdf.py piano_no_actuators.xml --output piano_with_keys.urdf
        """
    )
    parser.add_argument("mjcf_file", help="Path to the MJCF file")
    parser.add_argument("-o", "--output", help="Output URDF file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.mjcf_file).exists():
        print(f"Error: Input file '{args.mjcf_file}' does not exist")
        return

    try:
        converter = MjcfToUrdfConverter()
        output_path = converter.convert_to_urdf(args.mjcf_file, args.output)

        if args.verbose:
            print("\nConversion completed successfully!")
            print(f"Input: {args.mjcf_file}")
            print(f"Output: {output_path}")
        return

    except Exception as e:
        print(f"Error during conversion: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return


if __name__ == "__main__":
    main()
