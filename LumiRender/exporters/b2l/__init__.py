# Blender Add-on Template
# Contributor(s): Aaron Powell (aaron@lunadigital.tv)
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from . import auto_load
from . import material_nodes
bl_info = {
    "name": "Blender-LumiRender Exporter",
    "description": "",
    "author": "zijian",
    "version": (0, 2),
    "blender": (3, 0, 0),
    "location": " View3D > UI ",
    "warning": "",  # used for warning icon and text in add-ons panel
    "wiki_url": "https://git.duowan.com/ouzijian/blender2luminous",
    "support": "COMMUNITY",
    "category": "Render",
}


auto_load.init()


def register():
    auto_load.register()


def unregister():
    auto_load.unregister()
