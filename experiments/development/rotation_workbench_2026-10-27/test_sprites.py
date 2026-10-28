import tempfile
import unittest
from pathlib import Path
from PIL import Image
from sprites import tile_box,mask_alpha,slot_rows


class SpriteTests(unittest.TestCase):
    def test_all_forty_indices_have_unique_fixed_tiles(self):
        boxes=[tile_box(i,10,8) for i in range(40)]
        self.assertEqual(len(set(boxes)),40)
        self.assertEqual(boxes[0],(0,0,10,8))
        self.assertEqual(boxes[39],(70,32,80,40))
        with self.assertRaises(ValueError):tile_box(40,10,8)

    def test_mask_background_is_transparent_not_opaque_black(self):
        image=Image.new('L',(2,1));image.putdata([0,255])
        rgba=mask_alpha(image)
        self.assertEqual(rgba.convert('RGBA').getpixel((0,0))[3],0)
        self.assertEqual(rgba.convert('RGBA').getpixel((1,0))[3],255)

    def test_missing_views_do_not_shift_later_indices(self):
        rows=slot_rows([{'view':'0','source':'a'},{'view':'39','source':'z'}])
        self.assertEqual(len(rows),40)
        self.assertIsNone(rows[1])
        self.assertEqual(rows[39]['source'],'z')
        with self.assertRaises(ValueError):slot_rows([{'view':'0'},{'view':'0'}])


if __name__=='__main__':unittest.main()
