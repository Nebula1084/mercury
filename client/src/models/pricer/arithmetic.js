import { load, add, alter, update, edit, save, cancel } from "./util.js"
export default {
    namespace: 'arithmetic',
    state: {
        rows: [],
        stash: [],
        columns: [
            'Volatility', 'Maturity', 'Strike', 'Interest', 'Step', 'Path', 'Control Variate', 'Stock Price 0',
        ],
        stockNum: 1
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