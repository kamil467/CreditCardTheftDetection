using CreditCardTheftDetector.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using System.Diagnostics;
using static MLModel1;

namespace CreditCardTheftDetector.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;
        private const string DATA_FILEPATH = @"C:\Kamil\creditcard.csv";
        public HomeController(ILogger<HomeController> logger)
        {
            _logger = logger;
        }

        public IActionResult Index()
        {
            return View();
        }

        public IActionResult Privacy()
        {
            return View();
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }

        [HttpGet]
        public IActionResult PredictorIndex()
        {
            ModelInput sampleData = CreateSingleDataSample(DATA_FILEPATH);
            return View(sampleData);
        }
        [HttpPost]
        public IActionResult PredictorIndex(ModelInput modelInput)
        {
            ModelOutput prediction = MLModel1.Predict(modelInput);
            ViewBag.Prediction = prediction;
            return View();
        }

        private static ModelInput CreateSingleDataSample(string dataFilePath)
        {
            // Create MLContext  
            MLContext mlContext = new MLContext();

            // Load dataset  
            IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: dataFilePath,
                                            hasHeader: true,
                                            separatorChar: ',',
                                            allowQuoting: true,
                                            allowSparse: false);

            // Use first line of dataset as model input  
            // You can replace this with new test data (hardcoded or from end-user application)  
            ModelInput sampleForPrediction = mlContext.Data.CreateEnumerable<ModelInput>(dataView, false)
                                                                        .First();
            return sampleForPrediction;
        }
    }
}