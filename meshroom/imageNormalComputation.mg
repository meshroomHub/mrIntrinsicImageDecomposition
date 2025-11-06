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
                -298,
                -54
            ],
            "inputs": {}
        },
        "CopyFiles_1": {
            "nodeType": "CopyFiles",
            "position": [
                188,
                -43
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
                -47,
                -54
            ],
            "inputs": {
                "inputImages": "{CameraInit_1.output}",
                "outputDepth": false
            }
        }
    }
}