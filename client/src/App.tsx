import './App.css'
import Footer from './components/Footer'
import Header from './components/Header'
import { BrowserRouter } from 'react-router-dom'
import Routes from './components/RoutingLinks'

function App() {

  return (
    <BrowserRouter>
      <>
        <Header />
        <Routes />
        <Footer />
      </>
    </BrowserRouter>
  )
}

export default App
