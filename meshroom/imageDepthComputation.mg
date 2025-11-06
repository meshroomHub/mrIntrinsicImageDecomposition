{
    "header": {
        "releaseVersion": "2026.1.0+develop",
        "fileVersion": "2.0",
        "nodesVersions": {
            "CameraInit": "12.0",
            "CopyFiles": "1.3",
            "Marigold": "1.0"
        },
        "template": true
    },
    "graph": {
        "CameraInit_1": {
            "nodeType": "CameraInit",
            "position": [
                -262,
                -55
            ],
            "inputs": {}
        },
        "CopyFiles_1": {
            "nodeType": "CopyFiles",
            "position": [
                155,
                -45
            ],
            "inputs": {
                "inputFiles": "{Marigold_1.output}"
            }
        },
        "Marigold_1": {
            "nodeType": "Marigold",
            "position": [
                -54,
                -57
            ],
            "inputs": {
                "inputImages": "{CameraInit_1.output}",
                "computeNormals": false,
                "computeLighting": false,
                "computeAppearance": false
            }
        }
    }
}