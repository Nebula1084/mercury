import React from 'react'
import PropTypes from 'prop-types'
import { Router } from 'dva/router'
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
      getIndexRoute(nextState, cb) {
        require.ensure([], (require) => {
          registerModel(app, require('./models/dashboard'))
          cb(null, require('./routes/dashboard/'))
        }, 'dashboard')
      },
      childRoutes: [
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
          path: 'american',
          getComponent(nextState, cb) {
            require.ensure([], (require) => {
              registerModel(app, require('./models/pricer/american'))
              cb(null, require('./routes/american/'))
            }, 'american')
          }
        },
        {
          path: 'close',
          getComponent(nextState, cb) {
            require.ensure([], (require) => {
              registerModel(app, require('./models/pricer/close'))
              cb(null, require('./routes/close/'))
            }, 'close')
          }
        },
        {
          path: 'arithmetic',
          getComponent(nextState, cb) {
            require.ensure([], (require) => {
              registerModel(app, require('./models/pricer/arithmetic'))
              cb(null, require('./routes/arithmetic/'))
            }, 'arithmetic')
          }
        },
        {
          path: 'geometric',
          getComponent(nextState, cb) {
            require.ensure([], (require) => {
              registerModel(app, require('./models/pricer/geometric'))
              cb(null, require('./routes/geometric/'))
            }, 'geometric')
          }
        },
        {
          path: 'dashboard',
          getComponent(nextState, cb) {
            require.ensure([], (require) => {
              registerModel(app, require('./models/dashboard'))
              cb(null, require('./routes/dashboard/'))
            }, 'dashboard')
          }
        },
        {
          path: 'charts',
          getComponent(nextState, cb) {
            require.ensure([], (require) => {
              registerModel(app, require('./models/charts'))
              cb(null, require('./routes/charts/'))
            }, 'charts')
          }
        }
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
