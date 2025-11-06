{
    "header": {
        "releaseVersion": "2026.1.0+develop",
        "fileVersion": "2.0",
        "nodesVersions": {
            "CameraInit": "12.0",
            "CopyFiles": "1.3",
            "DepthAnythingV2": "1.0",
            "DepthPro": "1.0",
            "Marigold": "1.0",
            "MoGe": "1.0",
            "PixelPerfectDepth": "1.0",
            "StableNormal": "1.0"
        },
        "template": true
    },
    "graph": {
        "CameraInit_1": {
            "nodeType": "CameraInit",
            "position": [
                -294,
                98
            ],
            "inputs": {}
        },
        "CopyFiles_1": {
            "nodeType": "CopyFiles",
            "position": [
                536,
                74
            ],
            "inputs": {
                "inputFiles": [
                    "{MoGe_1.output}",
                    "{Marigold_1.output}",
                    "{StableNormal_1.output}",
                    "{DepthAnythingV2_1.output}",
                    "{DepthPro_1.output}",
                    "{PixelPerfectDepth_1.output}"
                ]
            }
        },
        "DepthAnythingV2_1": {
            "nodeType": "DepthAnythingV2",
            "position": [
                280,
                -81
            ],
            "inputs": {
                "inputImages": "{CameraInit_1.output}",
                "metricModel": false,
                "saveVisuImages": true
            }
        },
        "DepthPro_1": {
            "nodeType": "DepthPro",
            "position": [
                -19,
                285
            ],
            "inputs": {
                "inputImages": "{CameraInit_1.output}",
                "saveVisuImages": true
            }
        },
        "Marigold_1": {
            "nodeType": "Marigold",
            "position": [
                -20,
                18
            ],
            "inputs": {
                "inputImages": "{CameraInit_1.output}",
                "computeLighting": false,
                "computeAppearance": false,
                "saveVisuImages": true
            }
        },
        "MoGe_1": {
            "nodeType": "MoGe",
            "position": [
                -14,
                -141
            ],
            "inputs": {
                "inputImages": "{CameraInit_1.output}",
                "saveVisuImages": true
            }
        },
        "PixelPerfectDepth_1": {
            "nodeType": "PixelPerfectDepth",
            "position": [
                262,
                275
            ],
            "inputs": {
                "inputImages": "{CameraInit_1.output}",
                "saveVisuImages": true
            }
        },
        "StableNormal_1": {
            "nodeType": "StableNormal",
            "position": [
                -27,
                180
            ],
            "inputs": {
                "inputImages": "{CameraInit_1.output}"
            }
        }
    }
}