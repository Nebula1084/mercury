import React from 'react'
import PropTypes from 'prop-types'
import { Router, Switch, Route, Redirect } from 'dva/router'
import App from './routes/app'

const registerModel = (app, model) => {
  if (!(app._models.filter(m => m.namespace === model.namespace).length === 1)) {
    app.model(model)
  }
}

const Routers = function ({ history, app }) {
  const routes = [
    {
      path: '/',
      component: App,
      getIndexRoute (nextState, cb) {
        require.ensure([], (require) => {
          cb(null, { component: require('./routes/introduction/') })
        }, 'introduction')
      },
      childRoutes: [
        {
          path: 'introduction',
          getComponent(nextState, cb) {
            require.ensure([], (require) => {
              registerModel(app, require('./models/pricer/european'))
              cb(null, require('./routes/introduction/'))
            }, 'introduction')
          }
        },
        {
          path: 'european',
          getComponent(nextState, cb) {
            require.ensure([], (require) => {
              registerModel(app, require('./models/pricer/european'))
              cb(null, require('./routes/european/'))
            }, 'european')
          }
        },
        {
          path: 'volatility',
          getComponent(nextState, cb) {
            require.ensure([], (require) => {
              registerModel(app, require('./models/pricer/volatility'))
              cb(null, require('./routes/volatility/'))
            }, 'volatility')
          }
        },
        {
          path: 'geoEuro',
          getComponent(nextState, cb) {
            require.ensure([], (require) => {
              registerModel(app, require('./models/pricer/geoEuro'))
              cb(null, require('./routes/geoEuro/'))
            }, 'geoEuro')
          }
        },
        {
          path: 'arithEuro',
          getComponent(nextState, cb) {
            require.ensure([], (require) => {
              registerModel(app, require('./models/pricer/arithEuro'))
              cb(null, require('./routes/arithEuro/'))
            }, 'arithEuro')
          }
        },
        {
          path: 'american',
          getComponent(nextState, cb) {
            require.ensure([], (require) => {
              registerModel(app, require('./models/pricer/american'))
              cb(null, require('./routes/american/'))
            }, 'american')
          }
        },
        {
          path: 'arithAsian',
          getComponent(nextState, cb) {
            require.ensure([], (require) => {
              registerModel(app, require('./models/pricer/arithAsian'))
              cb(null, require('./routes/arithAsian/'))
            }, 'arithAsian')
          }
        },
        {
          path: 'geoAsian',
          getComponent(nextState, cb) {
            require.ensure([], (require) => {
              registerModel(app, require('./models/pricer/geoAsian'))
              cb(null, require('./routes/geoAsian/'))
            }, 'geoAsian')
          }
        },
      ]
    },
    {
      path: '*',
      getComponent(nextState, cb) {
        require.ensure([], (require) => {
          cb(null, require('./routes/notfound/'))
        }, 'notfound')
      }
    }
  ]

  return <Router history={history} routes={routes} />
}

Routers.propTypes = {
  history: PropTypes.object,
  app: PropTypes.object
}

export default Routers
