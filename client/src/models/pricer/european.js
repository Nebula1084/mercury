import { load, add, alter, update, edit, save, cancel } from "./util.js"
import { pricing, EUROPEAN, buildProtocol } from '../../services/price'

export default {
    namespace: 'european',
    state: {
        rows: [],
        stash: [],
        columns: [
            'Stock Price 0', 'Volatility 0', 'Maturity', 'Strike', 'Interest', 'Repo'
        ],
        stockNum: 1
    },
    effects: {
        *price({ payload: index }, { call, put, select }) {
            const data = yield select(state => state.european.rows[index])
            const protocol = buildProtocol(EUROPEAN, data)

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