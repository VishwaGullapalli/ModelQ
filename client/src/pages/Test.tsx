import axios from 'axios';
import { SetStateAction, useState } from 'react';

const Test = () => {
    const [algorithm, setAlgorithm] = useState('none');
    const [testCase, setTestCase] = useState('');
    const [serverResponse, setServerResponse] = useState(null);
    const handleAlgorithmChange = (e: { target: { value: SetStateAction<string>; }; }) => {
        setAlgorithm(e.target.value);
    };

    const handleTestCaseChange = (e: { target: { value: SetStateAction<string>; }; }) => {
        setTestCase(e.target.value);
    };  

    const handleSubmit = async (e: { preventDefault: () => void; }) => {
        e.preventDefault();

        // Check if algorithm and test case are selected/entered
        if (algorithm === 'none' || testCase.trim() === '') {
            alert('Please select an algorithm and enter a test case.');
            return;
        }

        try {
            const response = await axios.post(`http://localhost:5000/api/${algorithm}`, {
                testCase,
            });

            setServerResponse(response.data);
        } catch (error) {
            console.error('Error sending data to server:', error);
            // Handle the error as needed
            alert('Error sending data to server.');
        }
    };

    return (
        <section className="bg-gray-50 dark:bg-gray-900">
            <div className="flex flex-col items-center justify-center px-6 py-8 mx-auto md:h-screen lg:py-0">
                <a href="#" className="flex items-center mb-6 text-2xl font-semibold text-gray-900 dark:text-white">
                    <img className="w-8 h-8 mr-2" src="https://flowbite.s3.amazonaws.com/blocks/marketing-ui/logo.svg" alt="logo" />
                    Quantum
                </a>
                <div className="w-full bg-white rounded-lg shadow dark:border md:mt-0 sm:max-w-md xl:p-0 dark:bg-gray-800 dark:border-gray-700">
                    <div className="p-6 space-y-4 md:space-y-6 items-center sm:p-8">
                        <h1 className="text-xl font-bold leading-tight tracking-tight text-gray-900 md:text-2xl dark:text-white">
                            Test Quantum Algorithms
                        </h1>
                        <form className="space-y-4 md:space-y-6" onSubmit={handleSubmit}>
                            <div>
                                <label htmlFor="algorithm" className="block mb-2 text-sm font-medium text-gray-900 dark:text-white">Select the Algorithm</label>
                                {/* <input type="email" name="email" id="email" className="bg-gray-50 border border-gray-300 text-gray-900 sm:text-sm rounded-lg focus:ring-primary-600 focus:border-primary-600 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500" placeholder="name@company.com" required={true} /> */}
                                <select name="algorithm" id="algorithm" onChange={handleAlgorithmChange} className="bg-gray-50 border border-gray-300 text-gray-900 sm:text-sm rounded-lg focus:ring-primary-600 focus:border-primary-600 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500" required={true}>
                                    <option value="none" selected disabled hidden>Select an Option</option>
                                    <option value="shor">Shor's Algorithm</option>
                                    <option value="grovers">Grover's Algorithm</option>
                                </select>
                            </div>
                            <div>
                                <label htmlFor="text" className="block mb-2 text-sm font-medium text-gray-900 dark:text-white">Input Data</label>
                                <input type="text" name="text" id="testcase" onChange={handleTestCaseChange} placeholder="Test Case" className="bg-gray-50 border border-gray-300 text-gray-900 sm:text-sm rounded-lg focus:ring-primary-600 focus:border-primary-600 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500" required={true} />
                            </div>
                            {/* <div>
                                <label htmlFor="confirm-password" className="block mb-2 text-sm font-medium text-gray-900 dark:text-white">Confirm password</label>
                                <input type="confirm-password" name="confirm-password" id="confirm-password" placeholder="••••••••" className="bg-gray-50 border border-gray-300 text-gray-900 sm:text-sm rounded-lg focus:ring-primary-600 focus:border-primary-600 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500" required={true} />
                            </div> */}
                            {/* <div className="flex items-start">
                                <div className="flex items-center h-5">
                                    <input id="terms" aria-describedby="terms" type="checkbox" className="w-4 h-4 border border-gray-300 rounded bg-gray-50 focus:ring-3 focus:ring-primary-300 dark:bg-gray-700 dark:border-gray-600 dark:focus:ring-primary-600 dark:ring-offset-gray-800" required={true} />
                                </div>
                                <div className="ml-3 text-sm">
                                    <label htmlFor="terms" className="font-light text-gray-500 dark:text-gray-300">I accept the <a className="font-medium text-primary-600 hover:underline dark:text-primary-500" href="#">Terms and Conditions</a></label>
                                </div>
                            </div> */}
                            <button type="submit" className="w-full text-white bg-primary-600 hover:bg-primary-700 focus:ring-4 focus:outline-none focus:ring-primary-300 font-medium rounded-lg text-sm px-5 py-2.5 text-center dark:bg-primary-600 dark:hover:bg-primary-700 dark:focus:ring-primary-800">Send data to Server</button>
                            {/* <p className="text-sm font-light text-gray-500 dark:text-gray-400">
                                Already have an account? <a href="#" className="font-medium text-primary-600 hover:underline dark:text-primary-500">Login here</a>
                            </p> */}
                        </form>
                        {/* Render server response */}
                        {/* {serverResponse && (
                            <div className="mt-4 p-4 bg-green-100 dark:bg-green-700 text-green-900 dark:text-green-50 rounded-lg">
                                <p className="font-semibold">Server Response:</p>
                                <pre>{JSON.stringify(serverResponse, null, 2)}</pre>
                            </div>
                        )} */}
                        {serverResponse && (
                            <div className="mt-4 p-4 bg-green-100 dark:bg-green-700 text-green-900 dark:text-green-50 rounded-lg">
                                <p className="font-semibold">Server Response:</p>
                                {/* <ul>
                                    <li>{JSON.stringify(serverResponse, null, 2)}</li>{/*.statement}</li>
                                    <li>{serverResponse.p}</li>
                                    <li>{serverResponse.q}</li>
                                    <li>{serverResponse.time_taken}</li>
                                    <li>{serverResponse.total_time_taken}</li>
                                </ul> */}
                                {Object.keys(serverResponse).map((key) => (
                                    <li key={key}>
                                        {serverResponse[key]}
                                    </li>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </section>
    )
}

export default Test