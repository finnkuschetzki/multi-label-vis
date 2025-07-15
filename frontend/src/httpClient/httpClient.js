import axios from "axios"

const httpClient = axios.create({
    baseURL: "http://localhost:5000",
    timeout: 10000
})

export default httpClient
