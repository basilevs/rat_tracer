from pathlib import Path
from pprint import pprint
from typing import Iterator, Tuple
from xml.etree import ElementTree


def extract_box_coordinates(xml_path: Path) -> Iterator[Tuple[int, float, float, float, float]]:
    """
    Parse an XML file and yield rectangle coordinates from <box> elements under <track>.

    Args:
        xml_path: Path to the XML file.

    Yields:
        Tuple of floats: (xtl, ytl, xbr, ybr)
    """
    tree = ElementTree.parse(xml_path)
    root = tree.getroot()

    for box in root.findall('./track[@label="labyrinth"]/polygon'):
        try:
            if box.get('occluded') == '1':
                continue
            if box.get('outside') == '1':
                continue
            points = box.get("points").split(';')
            minx = 10000
            maxx = 0
            miny = 10000
            maxy = 0
            for p in points:
                x,y = map(float, p.split(','))
                minx = min(x, minx)
                miny = min(y, miny)
                maxx = max(x, maxx)
                maxy = max(y, maxy)

            coords = (
                int(box.get('frame')),
                float((minx + maxx) / 852 / 2),
                float((miny + maxy) / 480 / 2),
                float((maxx - minx)/ 852 ),
                float((maxy - miny)/ 480 ),
            )
            yield coords
        except (TypeError, ValueError):
            # Skip boxs with missing or non-float attributes
            continue


root = Path('/Users/vasiligulevich/git/rat_tracer/data/labels/')
for i in extract_box_coordinates(Path('input/annotations.xml')):
    idx = i[0]
    filename=f"frame_{idx:0>6}.txt"
    for p in [root / 'Train' / filename, root / 'Val' / filename]:
        if p.exists():
            print(p)
            print(p.read_text(), end='')
            print(2, *map(lambda x: f"{x:.{6}f}", i[1:]), sep=" ")
            with open(p, 'a', encoding='utf-8') as f:
                print(2, *map(lambda x: f"{x:.{6}f}", i[1:]), sep=" ", file=f)
