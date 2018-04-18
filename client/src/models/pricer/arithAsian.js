import { load, add, alter, update, edit, save, cancel } from "./util.js"
import { pricing, ARITHMETIC_ASIAN, buildProtocol } from '../../services/price'

export default {
    namespace: 'arithAsian',
    state: {
        rows: [],
        stash: [],
        columns: [
            'Stock Price 0', 'Volatility 0', 'Maturity', 'Strike', 'Interest', 'Step', 'Path',
        ],
        stockNum: 1
    },
    effects: {
        *price({ payload: index }, { call, put, select }) {
            const data = yield select(state => state.arithAsian.rows[index])
            const protocol = buildProtocol(ARITHMETIC_ASIAN, data)
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