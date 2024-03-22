/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "model.h"
#include "loss.h"
#include "utils.h"

#include <algorithm>
#include <stdexcept>

namespace fasttext {

Model::State::State(int32_t hiddenSize, int32_t outputSize, int32_t seed)
    : lossValue_(0.0),
      nexamples_(0),
      hidden(hiddenSize),
      output(outputSize),
      grad(hiddenSize),
      rng(seed) {}

real Model::State::getLoss() const {
  return lossValue_ / nexamples_;
}

void Model::State::incrementNExamples(real loss) {
  lossValue_ += loss;
  nexamples_++;
}

Model::Model(
    std::shared_ptr<Matrix> wi,
    std::shared_ptr<Matrix> wo,
    std::shared_ptr<Loss> loss,
    bool normalizeGradient)
    : wi_(wi), wo_(wo), loss_(loss), normalizeGradient_(normalizeGradient) {}

void Model::computeHidden(const std::vector<int32_t>& input, State& state)
    const {
  Vector& hidden = state.hidden;
  hidden.zero();
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    hidden.addRow(*wi_, *it);
  }
  hidden.mul(1.0 / input.size());
}

void Model::predict(
    const std::vector<int32_t>& input,
    int32_t k,
    real threshold,
    Predictions& heap,
    State& state) const {
  if (k == Model::kUnlimitedPredictions) {
    k = wo_->size(0); // output size
  } else if (k <= 0) {
    throw std::invalid_argument("k needs to be 1 or higher!");
  }
  heap.reserve(k + 1);
  computeHidden(input, state);

  loss_->predict(k, threshold, heap, state);
}

void Model::update(
    const std::vector<int32_t>& input,
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    real lr,
    State& state) {
  if (input.size() == 0) {
    return;
  }
  computeHidden(input, state);

  Vector& grad = state.grad;
  grad.zero();
  real lossValue = loss_->forward(targets, targetIndex, state, lr, true);
  state.incrementNExamples(lossValue);

  if (normalizeGradient_) {
    grad.mul(1.0 / input.size());
  }
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    wi_->addVectorToRow(grad, *it, 1.0);
  }
}

real Model::std_log(real x) const {
  return std::log(x + 1e-5);
}

void Model::save_chk(std::ostream& out, const Model::State* s, const Model* m) {

    s->hidden.save(out);
    s->output.save(out);
    s->grad.save(out);

    int32_t hsz = s->hidden.size();
    int32_t osz = s->output.size();

    out.write((char*)&hsz, sizeof(int32_t));   // TODO: remove this
    out.write((char*)&osz, sizeof(int32_t));   // TODO: remove this
    out.write((char*)&(s->lossValue_), sizeof(real));
    out.write((char*)&(s->nexamples_), sizeof(int64_t));

    int64_t n = m->loss_->t_sigmoid_.size();
    out.write((char*)&n, sizeof(int64_t));
    out.write((char*)m->loss_->t_sigmoid_.data(), n * sizeof(real));

    n = m->loss_->t_log_.size();
    out.write((char*)&n, sizeof(int64_t));
    out.write((char*)m->loss_->t_log_.data(), n * sizeof(real));
}

void Model::load_chk(std::istream& in, Model::State* s, Model* m) {

    s->hidden.load(in);
    s->output.load(in);
    s->grad.load(in);

    int32_t hsz = 0;
    int32_t osz = 0;

    in.read((char*)&hsz, sizeof(int32_t));
    in.read((char*)&osz, sizeof(int32_t));
    in.read((char*)&(s->lossValue_), sizeof(real));
    in.read((char*)&(s->nexamples_), sizeof(int64_t));

    int64_t n = 0;
    in.read((char*)&n, sizeof(int64_t));
    if (m->loss_->t_sigmoid_.size() != n) {
        m->loss_->t_sigmoid_.resize(n);
    }
    in.read((char*)m->loss_->t_sigmoid_.data(), n * sizeof(real));

    in.read((char*)&n, sizeof(int64_t));
    if (m->loss_->t_log_.size() != n) {
        m->loss_->t_log_.resize(n);
    }
    in.read((char*)m->loss_->t_log_.data(), n * sizeof(real));
}


} // namespace fasttext
