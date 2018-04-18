import { load, add, alter, update, edit, save, cancel } from "./util.js"
import { pricing, VOLATILITY, buildProtocol } from '../../services/price'

export default {
    namespace: 'volatility',
    state: {
        rows: [],
        stash: [],
        columns: [
            'Premium', 'Maturity', 'Strike', 'Interest', 'Repo', 'Stock Price 0',
        ],
        stockNum: 1
    },
    effects: {
        *price({ payload: index }, { call, put, select }) {
            const data = yield select(state => state.volatility.rows[index])
            const protocol = buildProtocol(VOLATILITY, data)
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