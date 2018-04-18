import { load, add, alter, update, edit, save, cancel } from "./util.js"
import { pricing, ARITHMETIC_EUROPEAN, buildProtocol } from '../../services/price'

export default {
    namespace: 'arithEuro',
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
            const data = yield select(state => state.arithEuro.rows[index])
            const protocol = buildProtocol(ARITHMETIC_EUROPEAN, data)
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