const path = require('path');

const webpack = require('webpack');
const autoprefixer = require('autoprefixer');
const precss = require('precss');
const functions = require('postcss-functions');
const ExtractTextPlugin = require('extract-text-webpack-plugin');

const postCssLoader = [
  'css-loader?modules',
  '&localIdentName=[name]__[local]___[hash:base64:5]',
  '&disableStructuralMinification',
  '!postcss-loader'
];

let plugins = [
  new webpack.NoErrorsPlugin(),
  new webpack.optimize.DedupePlugin(),
  new ExtractTextPlugin('bundle.css'),
];

if (process.env.NODE_ENV === 'production') {
  plugins = plugins.concat([
    new webpack.optimize.UglifyJsPlugin({
      output: {comments: false},
      test: /bundle\.js?$/
    }),
    new webpack.DefinePlugin({
      'process.env': {NODE_ENV: JSON.stringify('production')}
    })
  ]);

  postCssLoader.splice(1, 1); // drop human readable names
}


const config = {
  entry: {
    bundle: path.join(__dirname, 'client/index.js')
  },
  output: {
    path: path.join(__dirname, 'server/data/static/build'),
    publicPath: '/static/build/',
    filename: '[name].js'
  },
  plugins: plugins,
  module: {
    rules: [{
      test: /\.less$/,
      use: [{
        loader: 'style-loader' // creates style nodes from JS strings
      }, {
        loader: 'css-loader' // translates CSS into CommonJS
      }, {
        loader: 'less-loader', // compiles Less to CSS
      }]
    }],
    loaders: [
      {test: /\.css/, loader: ExtractTextPlugin.extract('style-loader', postCssLoader.join(''))},
      {test: /\.(png|gif)$/, loader: 'url-loader?name=[name]@[hash].[ext]&limit=5000'},
      {test: /\.svg$/, loader: 'url-loader?name=[name]@[hash].[ext]&limit=5000!svgo-loader?useConfig=svgo1'},
      {test: /\.(pdf|ico|jpg|eot|otf|woff|ttf|mp4|webm)$/, loader: 'file-loader?name=[name]@[hash].[ext]'},
      {test: /\.json$/, loader: 'json-loader'},
      {test: /\.less$/, loader: 'style-loader!css-loader!less-loader',},
      {
        test: /\.jsx?$/,
        include: path.join(__dirname, 'client'),
        loader: 'babel',
        query: {
          plugins: [
            ['import', {libraryName: 'antd', style: 'css'}]
          ]
        }
      }
    ]
  },
  resolve: {
    extensions: ['', '.js', '.jsx', '.css'],
    alias: {
      '#app': path.join(__dirname, 'client'),
      '#c': path.join(__dirname, 'client/components'),
      '#css': path.join(__dirname, 'client/css')
    }
  },
  svgo1: {
    multipass: true,
    plugins: [
      // by default enabled
      {mergePaths: false},
      {convertTransform: false},
      {convertShapeToPath: false},
      {cleanupIDs: false},
      {collapseGroups: false},
      {transformsWithOnePath: false},
      {cleanupNumericValues: false},
      {convertPathData: false},
      {moveGroupAttrsToElems: false},
      // by default disabled
      {removeTitle: true},
      {removeDesc: true}
    ]
  },
  postcss: function () {
    return [
      autoprefixer,
      precss({
        variables: {
          variables: require('./client/css/vars')
        }
      }),
      functions({
        functions: require('./client/css/funcs')
      })
    ];
  }
};

module.exports = config;
