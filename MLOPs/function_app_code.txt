import azure.functions as func
import logging
import pickle
import json


app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="Trigger")
def Trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    
    # Load the pickled SARIMA model
    with open('trained_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Make predictions for 31 days using the reopened model
    pred_uc = model.get_forecast(steps=31)
    sales_pred = pred_uc.predicted_mean.to_frame(name='Predicted Sales').values.tolist()
    sales_pred_truncated = [round(value[0], 2) for value in sales_pred]
    pred_ci = pred_uc.conf_int().values.tolist()
    pred_ci_truncated = [round(value[0], 2) for value in pred_ci]
    
    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        data = {
            "name": name,
            "sales_forecast": sales_pred_truncated,
            "confidence_interval": pred_ci_truncated
        }
        json_data = json.dumps(data)
        
        return func.HttpResponse(json_data, mimetype="application/json")
    else:
        return func.HttpResponse(
            "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
            status_code=200
        )
