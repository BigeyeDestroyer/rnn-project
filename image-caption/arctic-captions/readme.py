"""
Inputs for _step
(1) m_: (batch, n_steps)
(2) x_: (batch, 4 * hid_size), the 'Ey_{t - 1}' in equation [1]
(3) h_: (batch, hid_size)
(4) c_: (batch, hid_size)
(5) pctx_: (batch, ctxdims)
(6) dp_: (batch, 4 * hid_size) dropout mask



Outputs for _step
(1) h: (batch, hid_size)
(2) c: (batch, hid_size)
(3) alpha: (batch, L), each row the multinouilli params
(4) alpha_sample: (batch, L), each row the sampled results from multinouilli dist
(5) ctx_: (batch, ctxdim), context in equation [1]
(6) pstate_: (batch, ctxdim), hidden states projected to context space
             a part of equation [4]
(7) pctx_: (batch, ctxdim), context projected to context space
           and after adding, indicates the value before 'f_att' in equation [4]
(8) i : (batch, hid_size)
(9) f: (batch, hid_size)
(10) o: (batch, hid_size)
(11) preact: (batch, 4 * hid_size), values before T_{D + m + n, n} in equation [1]
(12) alpha_pre: (batch, L) alpha value before softmax
(13) pctx_list: list of an array, before affine in equation [4]
(14)

Others
(1) sel_: (batch, ), [0, 1] value for each sample, acting as suppression
(2) state_below: (batch, 4 * hid_size)
"""