import { load, add, alter, update, edit, save, cancel } from "./util.js"
import { pricing, AMERICAN, buildProtocol } from '../../services/price'

export default {
    namespace: 'american',
    state: {
        rows: [],
        stash: [],
        columns: [
            'Volatility', 'Maturity', 'Strike', 'Interest', 'Step', 'Stock Price 0',
        ],
        stockNum: 1
    },
    effects: {
        *price({ payload: index }, { call, put, select }) {
            const data = yield select(state => state.american.rows[index])
            const protocol = buildProtocol(AMERICAN, data)
            const result = yield call(pricing, protocol)
            yield put({ type: 'update', payload: { value: result.Mean, index: index, column: 'price' } })
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