import React, { PropTypes } from 'react'
import { connect } from 'dva'

export default class Introduction extends React.Component {

    render() {
        return (
            <div>
                <h1>Introduction</h1>
                <h2>Architecture</h2>
                <p>
                    Implement Front-end with React, which will be packed as static files. <br />
                    Server is implemented by Go and host the static files and HTTP API with Echo <br />
                    In order to take advantage of GPU, we implement pricer with C++ and CUDA. <br />
                    Go sever communicate with pricer by Unix Socket IPC. <br />
                </p>
                <h2>Monte-Carlo with GPU</h2>
                <img src={require('../../assets/img/MonteCarlo.png')} />
                <p>
                    Allocate sampling task and memory for every thread, so that they can sampling in parallel.O(N/P) <br />
                    1. Cholesky Decomposition <br />
                    2. Multi Thread Sampling <br />
                    3. Sum Reduction <br />
                    Every two elements add together. Only O(Log(N)) time complexity with sufficient processor. <br />
                    O(N/P+Log(P)) P is GPU core number
                    </p>
                <h2>Binomial Tree with GPU</h2>
                <img src={require('../../assets/img/Binomial.png')} />
                <p>
                    1. Multi Thread Iteration <br />
                    Improve to O(N) ?

    NO, O(N^2/P) <br />
                    2. Double buffer swap <br />
                    A[i]=F(A[i],A[i+1]) <br />
                    Such synchronization method only exists within one block.
    To avoid synchronization <br />
                    B[i]=F(A[i],A[i+1]) <br />
                    swap the pointer after each iteration. <br />
                </p>
            </div>
        )
    }
}