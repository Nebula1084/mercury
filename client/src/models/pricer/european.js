import { load, add, alter, update, edit, save, cancel } from "./util.js"
import { query } from '../../services/price'
export default {
    namespace: 'european',
    state: {
        rows: [],
        stash: [],
        columns: [
            'Volatility', 'Maturity', 'Strike', 'Interest', 'Repo', 'Stock Price 0',
        ],
        stockNum: 1
    },
    effects: {
        *price({ payload: index }, { call, put }) {
            const result = yield call(query, '/api/price')
            yield put({ type: 'update', payload: { value: result.price, index: index, column: 'price' } })
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