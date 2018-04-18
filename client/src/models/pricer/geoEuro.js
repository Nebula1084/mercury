import { load, add, alter, update, edit, save, cancel } from "./util.js"
import { pricing, GEOMETRIC_EUROPEAN, buildProtocol } from '../../services/price'

export default {
    namespace: 'geoEuro',
    state: {
        rows: [],
        stash: [],
        columns: [
            'Stock Price 0', 'Volatility 0', 'Maturity', 'Strike', 'Interest', 'PathNum'
        ],
        stockNum: 1
    },
    effects: {
        *price({ payload: index }, { call, put, select }) {
            const data = yield select(state => state.geoEuro.rows[index])
            data.basketSize = yield select(state => state.geoEuro.stockNum);
            const protocol = buildProtocol(GEOMETRIC_EUROPEAN, data)
            let t0 = performance.now();
            const result = yield call(pricing, protocol)
            let t1 = performance.now();
            yield put({ type: 'update', payload: { value: result.Mean, index: index, column: 'price' } })
            yield put({ type: 'update', payload: { value: result.Conf, index: index, column: 'conf' } })
            yield put({ type: 'update', payload: { value: t1 - t0, index: index, column: 'time' } })
        }
    },
    reducers: {
        import: load,
        add: add,
        alter, alter,
        update: update,
        edit: edit,
        save: save,
        cancel: cancel
    }
}