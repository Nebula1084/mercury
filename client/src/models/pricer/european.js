import { load, add, update, edit, save, cancel } from "./util.js"
export default {
    namespace: 'european',
    state: {
        rows: [],
        stash: []
    },
    reducers: {
        import: load,
        add: add,
        update: update,
        edit: edit,
        save: save,
        cancel: cancel
    }
}