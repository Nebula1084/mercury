export default {
  "entry": "src/index.js",
  "theme": "./theme.config.js",
  "outputPath": "../server/data",
  "env": {
    "development": {
      "extraBabelPlugins": [
        "transform-runtime",
        ["import", { "libraryName": "antd", "style": true }],
      ]
    },
    "production": {
      "extraBabelPlugins": [
        "transform-runtime",
        ["import", { "libraryName": "antd", "style": true }],
      ]
    }
  }
}
