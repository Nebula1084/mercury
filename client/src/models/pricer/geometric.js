import { load, add, alter, update, edit, save, cancel } from "./util.js"
import { routerRedux } from 'dva/router';

export default {
    namespace: 'geometric',
    state: {
        rows: [],
        stash: [],
        prices: [],
        columns: [
            'Volatility', 'Maturity', 'Strike', 'Interest', 'Step', 'Path', 'Control Variate', 'Stock Price 0',
        ],
        stockNum: 1,
        pricing: undefined
    },
    subscriptions: {
        setup({ history, dispatch }) {
            return history.listen(({ pathname, query }) => {
                if (pathname == '/monitor') {
                    let ws = new WebSocket("ws://localhost:5000/api/geometric");
                    function handler(event) {
                        dispatch({ type: 'newPrice', payload: event.data });
                    };
                    dispatch({ type: 'startPrice', payload: { connection: ws, index: query.index, handler: handler } });
                } else {
                    dispatch({ type: 'endPrice' });
                }
            });
        }
    },
    effects: {
        *price({ payload: index }, { call, put, select }) {
            yield put(routerRedux.push({
                pathname: '/monitor',
                query: { index: index },
            }));
        },
        *end() {

        }
    },
    reducers: {
        import: load,
        add: add,
        alter, alter,
        update: update,
        edit: edit,
        save: save,
        cancel: cancel,
        startPrice(state, { payload: { connection, index, handler } }) {
            if (state.connection == undefined) {
                console.log("start");
                state.connection = connection;
                state.connection.onmessage = handler;
                return { ...state, pricing: index, prices: [] };
            } else
                return state;
        },
        endPrice(state) {
            if (state.connection != undefined) {
                console.log("end");
                state.connection.close();
                return { ...state, pricing: undefined, connection: undefined };
            } else
                return state;
        },
        newPrice(state, { payload: data }) {
            state.prices.push(JSON.parse(data));
            return { ...state };
        }

    }
}