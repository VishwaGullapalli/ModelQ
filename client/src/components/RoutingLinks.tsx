import { Route, Routes } from 'react-router-dom'
import Algorithms from '../pages/Algorithms'
import Team from '../pages/Team'
import Home from '../pages/Home'
import Error from './Error'
import Maintenance from './Maintenance'
import Test from '../pages/Test'
import Contact from '../pages/Contact'

const RoutingLinks = () => {
  return (
    <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/algorithms" element={<Algorithms />} />
        <Route path="/team" element={<Team />} />
        <Route path="*" element={<Error />} />
        <Route path="/service" element={<Maintenance />} />
        <Route path="/test" element={<Test />} />
        <Route path="/contact" element={<Contact />} />
    </Routes>
  )
}

export default RoutingLinks