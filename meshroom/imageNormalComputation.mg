{
    "header": {
        "releaseVersion": "2026.1.0+develop",
        "fileVersion": "2.0",
        "nodesVersions": {
            "CameraInit": "12.0",
            "CopyFiles": "1.3",
            "MoGe": "1.0"
        },
        "template": true
    },
    "graph": {
        "CameraInit_1": {
            "nodeType": "CameraInit",
            "position": [
                0,
                0
            ],
            "inputs": {}
        },
        "CopyFiles_1": {
            "nodeType": "CopyFiles",
            "position": [
                400,
                0
            ],
            "inputs": {
                "inputFiles": [
                    "{MoGe_1.output}"
                ]
            }
        },
        "MoGe_1": {
            "nodeType": "MoGe",
            "position": [
                200,
                0
            ],
            "inputs": {
                "inputImages": "{CameraInit_1.output}",
                "outputDepth": false
            }
        }
    }
}