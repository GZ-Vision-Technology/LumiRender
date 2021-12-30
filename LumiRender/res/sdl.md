### material
 
    {
        "type": "MatteMaterial",
        "name": "matte_white",
        "param": {
            "color": [
                0.7250000238418579,
                0.7099999785423279,
                0.6800000071525574,
                1.0
            ]
        }
    },
    {
        "type": "MatteMaterial",
        "name": "matte_red",
        "param": {
            "color": [
                0.6299999952316284,
                0.06499999761581421,
                0.05000000074505806,
                1.0
            ]
        }
    },
    {
        "type": "MatteMaterial",
        "name": "matte_green",
        "param": {
            "color": [
                0.14000000059604645,
                0.44999998807907104,
                0.09099999815225601,
                1.0
            ]
        }
    },
    {
        "type": "MatteMaterial",
        "name": "matte_white_sigma",
        "param": {
            "color": [
                0.7250000238418579,
                0.7099999785423279,
                0.6800000071525574,
                1.0
            ],
            "sigma": 9.9
        }
    },
    {
        "type": "MatteMaterial",
        "name": "matte_white_sigma",
        "param": {
            "color": [
                0.7250000238418579,
                0.7099999785423279,
                0.6800000071525574,
                1.0
            ],
            "sigma": 9.9
        }
    },
    {
        "type": "MirrorMaterial",
        "name": "mirror_white",
        "param": {
            "color": [
                1,
                1,
                1,
                0
            ],
            "sigma": 9.9
        }
    },
    {
        "type": "GlassMaterial",
        "name": "glass_white",
        "param": {
            "color": [
                1,
                1,
                1,
                0
            ],
            "roughness" : [0.002,0.002],
            "eta": 1.9
        }
    },
    {
        "type": "GlassMaterial",
        "name": "rough_glass_white",
        "param": {
            "color": [
                1,
                1,
                1,
                0
            ],
            "roughness" : [0.002,0.002],
            "eta": 1.9
        }
    }
    ,
    {
        "type": "FakeMetalMaterial",
        "name" : "fake_metal",
        "param" : {
            "color" : [1,1,1],
            "roughness" : [0.05,0.05]
        }
    }
    ,
    {
        "type": "MetalMaterial",
        "name" : "metal",
        "param": {
            "eta": [0.19999069, 0.922084629, 1.09987593],
            "k" : [3.90463543, 2.44763327, 2.13765264],
            "roughness": [0.05,0.05]
        }
    }

### filter

{
    "type": "GaussianFilter",
    "param": {
        "radius": [0.5,0.5],
        "sigma" : 1.5
    }
},
{
    "type": "BoxFilter",
    "param": {
        "radius": [0.5,0.5]
    }
},
{
    "type": "TriangleFilter",
    "param": {
        "radius": [0.5,0.5]
    }
},
{
    "type": "MitchellFilter",
    "param": {
        "radius": [0.5,0.5],
        "B" : 0.33333,
        "C" : 0.33333
    }
},
{
    "type": "LanczosSincFilter",
    "param": {
        "radius": [0.5,0.5],
        "tau" : 1.5
    }
},