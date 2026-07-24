and  v27.16b, v26.16b, v5.16b
ushr v28.16b, v26.16b, #1
and  v28.16b, v28.16b, v5.16b
ushr v26.16b, v26.16b, #7
eor  v26.16b, v27.16b, v26.16b
tbl  v29.16b, { v1.16b, v2.16b, v3.16b, v4.16b }, v27.16b
tbl  v28.16b, { v16.16b, v17.16b, v18.16b, v19.16b }, v28.16b
tbl  v26.16b, { v20.16b, v21.16b, v22.16b, v23.16b }, v26.16b
