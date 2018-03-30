import { load, add, alter, update, edit, save, cancel } from "./util.js"
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