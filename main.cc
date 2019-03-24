#include <vector>
#include <chrono>
#include <iostream>
#include <iostream>
#include <fstream>


#include <re2/re2.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/string_util.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/tools/gen_op_registration.h>

#include "predictor.h"

void RegisterSelectedOps(tflite::MutableOpResolver*);

int
main(int argc, char const * argv[]) {
  std::ifstream in("backoff_response.txt");
  std::vector<std::string> backoff_response;
  std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("smartreply.tflite");

  std::string line;
  while (std::getline(in, line))
    backoff_response.push_back(line);
  tflite::custom::smartreply::SmartReplyConfig config(backoff_response);
  config.num_response = 1;
  config.backoff_confidence = 0;

  while (true) {
    std::cout << "> ";
    if (!std::getline(std::cin, line))
      continue;
    std::vector<tflite::custom::smartreply::PredictorResponse> predictor_responses;
    GetSegmentPredictions(
      {line},
      *model,
      config,
      &predictor_responses);

    if (predictor_responses.empty())
      continue;
    std::cout << predictor_responses.front().GetText() << std::endl;
  }
  return 0;
}

// vim:set cino=>2 et:
