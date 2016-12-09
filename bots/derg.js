(function isolate(){
    'use strict';
  var R = {}; // the Recurrent library

  (function(global) {
    "use strict";

    // Utility fun
    function assert(condition, message) {
      // from http://stackoverflow.com/questions/15313418/javascript-assert
      if (!condition) {
        message = message || "Assertion failed";
        if (typeof Error !== "undefined") {
          throw new Error(message);
        }
        throw message; // Fallback
      }
    }

    // Random numbers utils
    var return_v = false;
    var v_val = 0.0;
    var gaussRandom = function() {
      if(return_v) {
        return_v = false;
        return v_val;
      }
      var u = 2*Math.random()-1;
      var v = 2*Math.random()-1;
      var r = u*u + v*v;
      if(r == 0 || r > 1) return gaussRandom();
      var c = Math.sqrt(-2*Math.log(r)/r);
      v_val = v*c; // cache this
      return_v = true;
      return u*c;
    }
    var randf = function(a, b) { return Math.random()*(b-a)+a; }
    var randi = function(a, b) { return Math.floor(Math.random()*(b-a)+a); }
    var randn = function(mu, std){ return mu+gaussRandom()*std; }

    // helper function returns array of zeros of length n
    // and uses typed arrays if available
    var zeros = function(n) {
      if(typeof(n)==='undefined' || isNaN(n)) { return []; }
      if(typeof ArrayBuffer === 'undefined') {
        // lacking browser support
        var arr = new Array(n);
        for(var i=0;i<n;i++) { arr[i] = 0; }
        return arr;
      } else {
        return new Float64Array(n);
      }
    }

    // Mat holds a matrix
    var Mat = function(n,d) {
      // n is number of rows d is number of columns
      this.n = n;
      this.d = d;
      this.w = zeros(n * d);
      this.dw = zeros(n * d);
    }
    Mat.prototype = {
      get: function(row, col) {
        // slow but careful accessor function
        // we want row-major order
        var ix = (this.d * row) + col;
        assert(ix >= 0 && ix < this.w.length);
        return this.w[ix];
      },
      set: function(row, col, v) {
        // slow but careful accessor function
        var ix = (this.d * row) + col;
        assert(ix >= 0 && ix < this.w.length);
        this.w[ix] = v;
      },
      setFrom: function(arr) {
        for(var i=0,n=arr.length;i<n;i++) {
          this.w[i] = arr[i];
        }
      },
      setColumn: function(m, i) {
        for(var q=0,n=m.w.length;q<n;q++) {
          this.w[(this.d * q) + i] = m.w[q];
        }
      },
      toJSON: function() {
        var json = {};
        json['n'] = this.n;
        json['d'] = this.d;
        json['w'] = this.w;
        return json;
      },
      fromJSON: function(json) {
        this.n = json.n;
        this.d = json.d;
        this.w = zeros(this.n * this.d);
        this.dw = zeros(this.n * this.d);
        for(var i=0,n=this.n * this.d;i<n;i++) {
          this.w[i] = json.w[i]; // copy over weights
        }
      }
    }

    var copyMat = function(b) {
      var a = new Mat(b.n, b.d);
      a.setFrom(b.w);
      return a;
    }

    var copyNet = function(net) {
      // nets are (k,v) pairs with k = string key, v = Mat()
      var new_net = {};
      for(var p in net) {
        if(net.hasOwnProperty(p)){
          new_net[p] = copyMat(net[p]);
        }
      }
      return new_net;
    }

    var updateMat = function(m, alpha) {
      // updates in place
      for(var i=0,n=m.n*m.d;i<n;i++) {
        if(m.dw[i] !== 0) {
          m.w[i] += - alpha * m.dw[i];
          m.dw[i] = 0;
        }
      }
    }

    var updateNet = function(net, alpha) {
      for(var p in net) {
        if(net.hasOwnProperty(p)){
          updateMat(net[p], alpha);
        }
      }
    }

    var netToJSON = function(net) {
      var j = {};
      for(var p in net) {
        if(net.hasOwnProperty(p)){
          j[p] = net[p].toJSON();
        }
      }
      return j;
    }
    var netFromJSON = function(j) {
      var net = {};
      for(var p in j) {
        if(j.hasOwnProperty(p)){
          net[p] = new Mat(1,1); // not proud of this
          net[p].fromJSON(j[p]);
        }
      }
      return net;
    }
    var netZeroGrads = function(net) {
      for(var p in net) {
        if(net.hasOwnProperty(p)){
          var mat = net[p];
          gradFillConst(mat, 0);
        }
      }
    }
    var netFlattenGrads = function(net) {
      var n = 0;
      for(var p in net) { if(net.hasOwnProperty(p)){ var mat = net[p]; n += mat.dw.length; } }
      var g = new Mat(n, 1);
      var ix = 0;
      for(var p in net) {
        if(net.hasOwnProperty(p)){
          var mat = net[p];
          for(var i=0,m=mat.dw.length;i<m;i++) {
            g.w[ix] = mat.dw[i];
            ix++;
          }
        }
      }
      return g;
    }

    // return Mat but filled with random numbers from gaussian
    var RandMat = function(n,d,mu,std) {
      var m = new Mat(n, d);
      fillRandn(m,mu,std);
      //fillRand(m,-std,std); // kind of :P
      return m;
    }

    // Mat utils
    // fill matrix with random gaussian numbers
    var fillRandn = function(m, mu, std) { for(var i=0,n=m.w.length;i<n;i++) { m.w[i] = randn(mu, std); } }
    var fillRand = function(m, lo, hi) { for(var i=0,n=m.w.length;i<n;i++) { m.w[i] = randf(lo, hi); } }
    var gradFillConst = function(m, c) { for(var i=0,n=m.dw.length;i<n;i++) { m.dw[i] = c } }

    // Transformer definitions
    var Graph = function(needs_backprop) {
      if(typeof needs_backprop === 'undefined') { needs_backprop = true; }
      this.needs_backprop = needs_backprop;

      // this will store a list of functions that perform backprop,
      // in their forward pass order. So in backprop we will go
      // backwards and evoke each one
      this.backprop = [];
    }
    Graph.prototype = {
      backward: function() {
        for(var i=this.backprop.length-1;i>=0;i--) {
          this.backprop[i](); // tick!
        }
      },
      rowPluck: function(m, ix) {
        // pluck a row of m with index ix and return it as col vector
        assert(ix >= 0 && ix < m.n);
        var d = m.d;
        var out = new Mat(d, 1);
        for(var i=0,n=d;i<n;i++){ out.w[i] = m.w[d * ix + i]; } // copy over the data

        if(this.needs_backprop) {
          var backward = function() {
            for(var i=0,n=d;i<n;i++){ m.dw[d * ix + i] += out.dw[i]; }
          }
          this.backprop.push(backward);
        }
        return out;
      },
      tanh: function(m) {
        // tanh nonlinearity
        var out = new Mat(m.n, m.d);
        var n = m.w.length;
        for(var i=0;i<n;i++) {
          out.w[i] = Math.tanh(m.w[i]);
        }

        if(this.needs_backprop) {
          var backward = function() {
            for(var i=0;i<n;i++) {
              // grad for z = tanh(x) is (1 - z^2)
              var mwi = out.w[i];
              m.dw[i] += (1.0 - mwi * mwi) * out.dw[i];
            }
          }
          this.backprop.push(backward);
        }
        return out;
      },
      sigmoid: function(m) {
        // sigmoid nonlinearity
        var out = new Mat(m.n, m.d);
        var n = m.w.length;
        for(var i=0;i<n;i++) {
          out.w[i] = sig(m.w[i]);
        }

        if(this.needs_backprop) {
          var backward = function() {
            for(var i=0;i<n;i++) {
              // grad for z = tanh(x) is (1 - z^2)
              var mwi = out.w[i];
              m.dw[i] += mwi * (1.0 - mwi) * out.dw[i];
            }
          }
          this.backprop.push(backward);
        }
        return out;
      },
      relu: function(m) {
        var out = new Mat(m.n, m.d);
        var n = m.w.length;
        for(var i=0;i<n;i++) {
          out.w[i] = Math.max(0, m.w[i]); // relu
        }
        if(this.needs_backprop) {
          var backward = function() {
            for(var i=0;i<n;i++) {
              m.dw[i] += m.w[i] > 0 ? out.dw[i] : 0.0;
            }
          }
          this.backprop.push(backward);
        }
        return out;
      },
      mul: function(m1, m2) {
        // multiply matrices m1 * m2
        assert(m1.d === m2.n, 'matmul dimensions misaligned');

        var n = m1.n;
        var d = m2.d;
        var out = new Mat(n,d);
        for(var i=0;i<m1.n;i++) { // loop over rows of m1
          for(var j=0;j<m2.d;j++) { // loop over cols of m2
            var dot = 0.0;
            for(var k=0;k<m1.d;k++) { // dot product loop
              dot += m1.w[m1.d*i+k] * m2.w[m2.d*k+j];
            }
            out.w[d*i+j] = dot;
          }
        }

        if(this.needs_backprop) {
          var backward = function() {
            for(var i=0;i<m1.n;i++) { // loop over rows of m1
              for(var j=0;j<m2.d;j++) { // loop over cols of m2
                for(var k=0;k<m1.d;k++) { // dot product loop
                  var b = out.dw[d*i+j];
                  m1.dw[m1.d*i+k] += m2.w[m2.d*k+j] * b;
                  m2.dw[m2.d*k+j] += m1.w[m1.d*i+k] * b;
                }
              }
            }
          }
          this.backprop.push(backward);
        }
        return out;
      },
      add: function(m1, m2) {
        assert(m1.w.length === m2.w.length);

        var out = new Mat(m1.n, m1.d);
        for(var i=0,n=m1.w.length;i<n;i++) {
          out.w[i] = m1.w[i] + m2.w[i];
        }
        if(this.needs_backprop) {
          var backward = function() {
            for(var i=0,n=m1.w.length;i<n;i++) {
              m1.dw[i] += out.dw[i];
              m2.dw[i] += out.dw[i];
            }
          }
          this.backprop.push(backward);
        }
        return out;
      },
      dot: function(m1, m2) {
        // m1 m2 are both column vectors
        assert(m1.w.length === m2.w.length);
        var out = new Mat(1,1);
        var dot = 0.0;
        for(var i=0,n=m1.w.length;i<n;i++) {
          dot += m1.w[i] * m2.w[i];
        }
        out.w[0] = dot;
        if(this.needs_backprop) {
          var backward = function() {
            for(var i=0,n=m1.w.length;i<n;i++) {
              m1.dw[i] += m2.w[i] * out.dw[0];
              m2.dw[i] += m1.w[i] * out.dw[0];
            }
          }
          this.backprop.push(backward);
        }
        return out;
      },
      eltmul: function(m1, m2) {
        assert(m1.w.length === m2.w.length);

        var out = new Mat(m1.n, m1.d);
        for(var i=0,n=m1.w.length;i<n;i++) {
          out.w[i] = m1.w[i] * m2.w[i];
        }
        if(this.needs_backprop) {
          var backward = function() {
            for(var i=0,n=m1.w.length;i<n;i++) {
              m1.dw[i] += m2.w[i] * out.dw[i];
              m2.dw[i] += m1.w[i] * out.dw[i];
            }
          }
          this.backprop.push(backward);
        }
        return out;
      },
    }

    var softmax = function(m) {
        var out = new Mat(m.n, m.d); // probability volume
        var maxval = -999999;
        for(var i=0,n=m.w.length;i<n;i++) { if(m.w[i] > maxval) maxval = m.w[i]; }

        var s = 0.0;
        for(var i=0,n=m.w.length;i<n;i++) {
          out.w[i] = Math.exp(m.w[i] - maxval);
          s += out.w[i];
        }
        for(var i=0,n=m.w.length;i<n;i++) { out.w[i] /= s; }

        // no backward pass here needed
        // since we will use the computed probabilities outside
        // to set gradients directly on m
        return out;
      }

    var Solver = function() {
      this.decay_rate = 0.999;
      this.smooth_eps = 1e-8;
      this.step_cache = {};
    }
    Solver.prototype = {
      step: function(model, step_size, regc, clipval) {
        // perform parameter update
        var solver_stats = {};
        var num_clipped = 0;
        var num_tot = 0;
        for(var k in model) {
          if(model.hasOwnProperty(k)) {
            var m = model[k]; // mat ref
            if(!(k in this.step_cache)) { this.step_cache[k] = new Mat(m.n, m.d); }
            var s = this.step_cache[k];
            for(var i=0,n=m.w.length;i<n;i++) {

              // rmsprop adaptive learning rate
              var mdwi = m.dw[i];
              s.w[i] = s.w[i] * this.decay_rate + (1.0 - this.decay_rate) * mdwi * mdwi;

              // gradient clip
              if(mdwi > clipval) {
                mdwi = clipval;
                num_clipped++;
              }
              if(mdwi < -clipval) {
                mdwi = -clipval;
                num_clipped++;
              }
              num_tot++;

              // update (and regularize)
              m.w[i] += - step_size * mdwi / Math.sqrt(s.w[i] + this.smooth_eps) - regc * m.w[i];
              m.dw[i] = 0; // reset gradients for next iteration
            }
          }
        }
        solver_stats['ratio_clipped'] = num_clipped*1.0/num_tot;
        return solver_stats;
      }
    }

    var initLSTM = function(input_size, hidden_sizes, output_size) {
      // hidden size should be a list

      var model = {};
      for(var d=0;d<hidden_sizes.length;d++) { // loop over depths
        var prev_size = d === 0 ? input_size : hidden_sizes[d - 1];
        var hidden_size = hidden_sizes[d];

        // gates parameters
        model['Wix'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);
        model['Wih'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
        model['bi'+d] = new Mat(hidden_size, 1);
        model['Wfx'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);
        model['Wfh'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
        model['bf'+d] = new Mat(hidden_size, 1);
        model['Wox'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);
        model['Woh'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
        model['bo'+d] = new Mat(hidden_size, 1);
        // cell write params
        model['Wcx'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);
        model['Wch'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
        model['bc'+d] = new Mat(hidden_size, 1);
      }
      // decoder params
      model['Whd'] = new RandMat(output_size, hidden_size, 0, 0.08);
      model['bd'] = new Mat(output_size, 1);
      return model;
    }

    var forwardLSTM = function(G, model, hidden_sizes, x, prev) {
      // forward prop for a single tick of LSTM
      // G is graph to append ops to
      // model contains LSTM parameters
      // x is 1D column vector with observation
      // prev is a struct containing hidden and cell
      // from previous iteration

      if(prev == null || typeof prev.h === 'undefined') {
        var hidden_prevs = [];
        var cell_prevs = [];
        for(var d=0;d<hidden_sizes.length;d++) {
          hidden_prevs.push(new R.Mat(hidden_sizes[d],1));
          cell_prevs.push(new R.Mat(hidden_sizes[d],1));
        }
      } else {
        var hidden_prevs = prev.h;
        var cell_prevs = prev.c;
      }

      var hidden = [];
      var cell = [];
      for(var d=0;d<hidden_sizes.length;d++) {

        var input_vector = d === 0 ? x : hidden[d-1];
        var hidden_prev = hidden_prevs[d];
        var cell_prev = cell_prevs[d];

        // input gate
        var h0 = G.mul(model['Wix'+d], input_vector);
        var h1 = G.mul(model['Wih'+d], hidden_prev);
        var input_gate = G.sigmoid(G.add(G.add(h0,h1),model['bi'+d]));

        // forget gate
        var h2 = G.mul(model['Wfx'+d], input_vector);
        var h3 = G.mul(model['Wfh'+d], hidden_prev);
        var forget_gate = G.sigmoid(G.add(G.add(h2, h3),model['bf'+d]));

        // output gate
        var h4 = G.mul(model['Wox'+d], input_vector);
        var h5 = G.mul(model['Woh'+d], hidden_prev);
        var output_gate = G.sigmoid(G.add(G.add(h4, h5),model['bo'+d]));

        // write operation on cells
        var h6 = G.mul(model['Wcx'+d], input_vector);
        var h7 = G.mul(model['Wch'+d], hidden_prev);
        var cell_write = G.tanh(G.add(G.add(h6, h7),model['bc'+d]));

        // compute new cell activation
        var retain_cell = G.eltmul(forget_gate, cell_prev); // what do we keep from cell
        var write_cell = G.eltmul(input_gate, cell_write); // what do we write to cell
        var cell_d = G.add(retain_cell, write_cell); // new cell contents

        // compute hidden state as gated, saturated cell activations
        var hidden_d = G.eltmul(output_gate, G.tanh(cell_d));

        hidden.push(hidden_d);
        cell.push(cell_d);
      }

      // one decoder to outputs at end
      var output = G.add(G.mul(model['Whd'], hidden[hidden.length - 1]),model['bd']);

      // return cell memory, hidden representation and output
      return {'h':hidden, 'c':cell, 'o' : output};
    }

    var sig = function(x) {
      // helper function for computing sigmoid
      return 1.0/(1+Math.exp(-x));
    }

    var maxi = function(w) {
      // argmax of array w
      var maxv = w[0];
      var maxix = 0;
      for(var i=1,n=w.length;i<n;i++) {
        var v = w[i];
        if(v > maxv) {
          maxix = i;
          maxv = v;
        }
      }
      return maxix;
    }

    var samplei = function(w) {
      // sample argmax from w, assuming w are
      // probabilities that sum to one
      var r = randf(0,1);
      var x = 0.0;
      var i = 0;
      while(true) {
        x += w[i];
        if(x > r) { return i; }
        i++;
      }
      return w.length - 1; // pretty sure we should never get here?
    }

    // various utils
    global.assert = assert;
    global.zeros = zeros;
    global.maxi = maxi;
    global.samplei = samplei;
    global.randi = randi;
    global.randn = randn;
    global.softmax = softmax;
    // classes
    global.Mat = Mat;
    global.RandMat = RandMat;
    global.forwardLSTM = forwardLSTM;
    global.initLSTM = initLSTM;
    // more utils
    global.updateMat = updateMat;
    global.updateNet = updateNet;
    global.copyMat = copyMat;
    global.copyNet = copyNet;
    global.netToJSON = netToJSON;
    global.netFromJSON = netFromJSON;
    global.netZeroGrads = netZeroGrads;
    global.netFlattenGrads = netFlattenGrads;
    // optimization
    global.Solver = Solver;
    global.Graph = Graph;
  })(R);

  // END OF RECURRENTJS

  var RL = {};
  (function(global) {
    "use strict";

  // syntactic sugar function for getting default parameter values
  var getopt = function(opt, field_name, default_value) {
    if(typeof opt === 'undefined') { return default_value; }
    return (typeof opt[field_name] !== 'undefined') ? opt[field_name] : default_value;
  }

  var zeros = R.zeros; // inherit these
  var assert = R.assert;
  var randi = R.randi;
  var randf = R.randf;

  var setConst = function(arr, c) {
    for(var i=0,n=arr.length;i<n;i++) {
      arr[i] = c;
    }
  }

  var sampleWeighted = function(p) {
    var r = Math.random();
    var c = 0.0;
    for(var i=0,n=p.length;i<n;i++) {
      c += p[i];
      if(c >= r) { return i; }
    }
    assert(false, 'wtf');
  }

  // ------
  // AGENTS
  // ------

  // DPAgent performs Value Iteration
  // - can also be used for Policy Iteration if you really wanted to
  // - requires model of the environment :(
  // - does not learn from experience :(
  // - assumes finite MDP :(
  var DPAgent = function(env, opt) {
    this.V = null; // state value function
    this.P = null; // policy distribution \pi(s,a)
    this.env = env; // store pointer to environment
    this.gamma = getopt(opt, 'gamma', 0.75); // future reward discount factor
    this.reset();
  }
  DPAgent.prototype = {
    reset: function() {
      // reset the agent's policy and value function
      this.ns = this.env.getNumStates();
      this.na = this.env.getMaxNumActions();
      this.V = zeros(this.ns);
      this.P = zeros(this.ns * this.na);
      // initialize uniform random policy
      for(var s=0;s<this.ns;s++) {
        var poss = this.env.allowedActions(s);
        for(var i=0,n=poss.length;i<n;i++) {
          this.P[poss[i]*this.ns+s] = 1.0 / poss.length;
        }
      }
    },
    act: function(s) {
      // behave according to the learned policy
      var poss = this.env.allowedActions(s);
      var ps = [];
      for(var i=0,n=poss.length;i<n;i++) {
        var a = poss[i];
        var prob = this.P[a*this.ns+s];
        ps.push(prob);
      }
      var maxi = sampleWeighted(ps);
      return poss[maxi];
    },
    learn: function() {
      // perform a single round of value iteration
      self.evaluatePolicy(); // writes this.V
      self.updatePolicy(); // writes this.P
    },
    evaluatePolicy: function() {
      // perform a synchronous update of the value function
      var Vnew = zeros(this.ns);
      for(var s=0;s<this.ns;s++) {
        // integrate over actions in a stochastic policy
        // note that we assume that policy probability mass over allowed actions sums to one
        var v = 0.0;
        var poss = this.env.allowedActions(s);
        for(var i=0,n=poss.length;i<n;i++) {
          var a = poss[i];
          var prob = this.P[a*this.ns+s]; // probability of taking action under policy
          if(prob === 0) { continue; } // no contribution, skip for speed
          var ns = this.env.nextStateDistribution(s,a);
          var rs = this.env.reward(s,a,ns); // reward for s->a->ns transition
          v += prob * (rs + this.gamma * this.V[ns]);
        }
        Vnew[s] = v;
      }
      this.V = Vnew; // swap
    },
    updatePolicy: function() {
      // update policy to be greedy w.r.t. learned Value function
      for(var s=0;s<this.ns;s++) {
        var poss = this.env.allowedActions(s);
        // compute value of taking each allowed action
        var vmax, nmax;
        var vs = [];
        for(var i=0,n=poss.length;i<n;i++) {
          var a = poss[i];
          var ns = this.env.nextStateDistribution(s,a);
          var rs = this.env.reward(s,a,ns);
          var v = rs + this.gamma * this.V[ns];
          vs.push(v);
          if(i === 0 || v > vmax) { vmax = v; nmax = 1; }
          else if(v === vmax) { nmax += 1; }
        }
        // update policy smoothly across all argmaxy actions
        for(var i=0,n=poss.length;i<n;i++) {
          var a = poss[i];
          this.P[a*this.ns+s] = (vs[i] === vmax) ? 1.0/nmax : 0.0;
        }
      }
    },
  }

  // QAgent uses TD (Q-Learning, SARSA)
  // - does not require environment model :)
  // - learns from experience :)
  var TDAgent = function(env, opt) {
    this.update = getopt(opt, 'update', 'qlearn'); // qlearn | sarsa
    this.gamma = getopt(opt, 'gamma', 0.75); // future reward discount factor
    this.epsilon = getopt(opt, 'epsilon', 0.1); // for epsilon-greedy policy
    this.alpha = getopt(opt, 'alpha', 0.01); // value function learning rate

    // class allows non-deterministic policy, and smoothly regressing towards the optimal policy based on Q
    this.smooth_policy_update = getopt(opt, 'smooth_policy_update', false);
    this.beta = getopt(opt, 'beta', 0.01); // learning rate for policy, if smooth updates are on

    // eligibility traces
    this.lambda = getopt(opt, 'lambda', 0); // eligibility trace decay. 0 = no eligibility traces used
    this.replacing_traces = getopt(opt, 'replacing_traces', true);

    // optional optimistic initial values
    this.q_init_val = getopt(opt, 'q_init_val', 0);

    this.planN = getopt(opt, 'planN', 0); // number of planning steps per learning iteration (0 = no planning)

    this.Q = null; // state action value function
    this.P = null; // policy distribution \pi(s,a)
    this.e = null; // eligibility trace
    this.env_model_s = null;; // environment model (s,a) -> (s',r)
    this.env_model_r = null;; // environment model (s,a) -> (s',r)
    this.env = env; // store pointer to environment
    this.reset();
  }
  TDAgent.prototype = {
    reset: function(){
      // reset the agent's policy and value function
      this.ns = this.env.getNumStates();
      this.na = this.env.getMaxNumActions();
      this.Q = zeros(this.ns * this.na);
      if(this.q_init_val !== 0) { setConst(this.Q, this.q_init_val); }
      this.P = zeros(this.ns * this.na);
      this.e = zeros(this.ns * this.na);

      // model/planning vars
      this.env_model_s = zeros(this.ns * this.na);
      setConst(this.env_model_s, -1); // init to -1 so we can test if we saw the state before
      this.env_model_r = zeros(this.ns * this.na);
      this.sa_seen = [];
      this.pq = zeros(this.ns * this.na);

      // initialize uniform random policy
      for(var s=0;s<this.ns;s++) {
        var poss = this.env.allowedActions(s);
        for(var i=0,n=poss.length;i<n;i++) {
          this.P[poss[i]*this.ns+s] = 1.0 / poss.length;
        }
      }
      // agent memory, needed for streaming updates
      // (s0,a0,r0,s1,a1,r1,...)
      this.r0 = null;
      this.s0 = null;
      this.s1 = null;
      this.a0 = null;
      this.a1 = null;
    },
    resetEpisode: function() {
      // an episode finished
    },
    act: function(s){
      // act according to epsilon greedy policy
      var poss = this.env.allowedActions(s);
      var probs = [];
      for(var i=0,n=poss.length;i<n;i++) {
        probs.push(this.P[poss[i]*this.ns+s]);
      }
      // epsilon greedy policy
      if(Math.random() < this.epsilon) {
        var a = poss[randi(0,poss.length)]; // random available action
        this.explored = true;
      } else {
        var a = poss[sampleWeighted(probs)];
        this.explored = false;
      }
      // shift state memory
      this.s0 = this.s1;
      this.a0 = this.a1;
      this.s1 = s;
      this.a1 = a;
      return a;
    },
    learn: function(r1){
      // takes reward for previous action, which came from a call to act()
      if(!(this.r0 == null)) {
        this.learnFromTuple(this.s0, this.a0, this.r0, this.s1, this.a1, this.lambda);
        if(this.planN > 0) {
          this.updateModel(this.s0, this.a0, this.r0, this.s1);
          this.plan();
        }
      }
      this.r0 = r1; // store this for next update
    },
    updateModel: function(s0, a0, r0, s1) {
      // transition (s0,a0) -> (r0,s1) was observed. Update environment model
      var sa = a0 * this.ns + s0;
      if(this.env_model_s[sa] === -1) {
        // first time we see this state action
        this.sa_seen.push(a0 * this.ns + s0); // add as seen state
      }
      this.env_model_s[sa] = s1;
      this.env_model_r[sa] = r0;
    },
    plan: function() {

      // order the states based on current priority queue information
      var spq = [];
      for(var i=0,n=this.sa_seen.length;i<n;i++) {
        var sa = this.sa_seen[i];
        var sap = this.pq[sa];
        if(sap > 1e-5) { // gain a bit of efficiency
          spq.push({sa:sa, p:sap});
        }
      }
      spq.sort(function(a,b){ return a.p < b.p ? 1 : -1});

      // perform the updates
      var nsteps = Math.min(this.planN, spq.length);
      for(var k=0;k<nsteps;k++) {
        // random exploration
        //var i = randi(0, this.sa_seen.length); // pick random prev seen state action
        //var s0a0 = this.sa_seen[i];
        var s0a0 = spq[k].sa;
        this.pq[s0a0] = 0; // erase priority, since we're backing up this state
        var s0 = s0a0 % this.ns;
        var a0 = Math.floor(s0a0 / this.ns);
        var r0 = this.env_model_r[s0a0];
        var s1 = this.env_model_s[s0a0];
        var a1 = -1; // not used for Q learning
        if(this.update === 'sarsa') {
          // generate random action?...
          var poss = this.env.allowedActions(s1);
          var a1 = poss[randi(0,poss.length)];
        }
        this.learnFromTuple(s0, a0, r0, s1, a1, 0); // note lambda = 0 - shouldnt use eligibility trace here
      }
    },
    learnFromTuple: function(s0, a0, r0, s1, a1, lambda) {
      var sa = a0 * this.ns + s0;

      // calculate the target for Q(s,a)
      if(this.update === 'qlearn') {
        // Q learning target is Q(s0,a0) = r0 + gamma * max_a Q[s1,a]
        var poss = this.env.allowedActions(s1);
        var qmax = 0;
        for(var i=0,n=poss.length;i<n;i++) {
          var s1a = poss[i] * this.ns + s1;
          var qval = this.Q[s1a];
          if(i === 0 || qval > qmax) { qmax = qval; }
        }
        var target = r0 + this.gamma * qmax;
      } else if(this.update === 'sarsa') {
        // SARSA target is Q(s0,a0) = r0 + gamma * Q[s1,a1]
        var s1a1 = a1 * this.ns + s1;
        var target = r0 + this.gamma * this.Q[s1a1];
      }

      if(lambda > 0) {
        // perform an eligibility trace update
        if(this.replacing_traces) {
          this.e[sa] = 1;
        } else {
          this.e[sa] += 1;
        }
        var edecay = lambda * this.gamma;
        var state_update = zeros(this.ns);
        for(var s=0;s<this.ns;s++) {
          var poss = this.env.allowedActions(s);
          for(var i=0;i<poss.length;i++) {
            var a = poss[i];
            var saloop = a * this.ns + s;
            var esa = this.e[saloop];
            var update = this.alpha * esa * (target - this.Q[saloop]);
            this.Q[saloop] += update;
            this.updatePriority(s, a, update);
            this.e[saloop] *= edecay;
            var u = Math.abs(update);
            if(u > state_update[s]) { state_update[s] = u; }
          }
        }
        for(var s=0;s<this.ns;s++) {
          if(state_update[s] > 1e-5) { // save efficiency here
            this.updatePolicy(s);
          }
        }
        if(this.explored && this.update === 'qlearn') {
          // have to wipe the trace since q learning is off-policy :(
          this.e = zeros(this.ns * this.na);
        }
      } else {
        // simpler and faster update without eligibility trace
        // update Q[sa] towards it with some step size
        var update = this.alpha * (target - this.Q[sa]);
        this.Q[sa] += update;
        this.updatePriority(s0, a0, update);
        // update the policy to reflect the change (if appropriate)
        this.updatePolicy(s0);
      }
    },
    updatePriority: function(s,a,u) {
      // used in planning. Invoked when Q[sa] += update
      // we should find all states that lead to (s,a) and upgrade their priority
      // of being update in the next planning step
      u = Math.abs(u);
      if(u < 1e-5) { return; } // for efficiency skip small updates
      if(this.planN === 0) { return; } // there is no planning to be done, skip.
      for(var si=0;si<this.ns;si++) {
        // note we are also iterating over impossible actions at all states,
        // but this should be okay because their env_model_s should simply be -1
        // as initialized, so they will never be predicted to point to any state
        // because they will never be observed, and hence never be added to the model
        for(var ai=0;ai<this.na;ai++) {
          var siai = ai * this.ns + si;
          if(this.env_model_s[siai] === s) {
            // this state leads to s, add it to priority queue
            this.pq[siai] += u;
          }
        }
      }
    },
    updatePolicy: function(s) {
      var poss = this.env.allowedActions(s);
      // set policy at s to be the action that achieves max_a Q(s,a)
      // first find the maxy Q values
      var qmax, nmax;
      var qs = [];
      for(var i=0,n=poss.length;i<n;i++) {
        var a = poss[i];
        var qval = this.Q[a*this.ns+s];
        qs.push(qval);
        if(i === 0 || qval > qmax) { qmax = qval; nmax = 1; }
        else if(qval === qmax) { nmax += 1; }
      }
      // now update the policy smoothly towards the argmaxy actions
      var psum = 0.0;
      for(var i=0,n=poss.length;i<n;i++) {
        var a = poss[i];
        var target = (qs[i] === qmax) ? 1.0/nmax : 0.0;
        var ix = a*this.ns+s;
        if(this.smooth_policy_update) {
          // slightly hacky :p
          this.P[ix] += this.beta * (target - this.P[ix]);
          psum += this.P[ix];
        } else {
          // set hard target
          this.P[ix] = target;
        }
      }
      if(this.smooth_policy_update) {
        // renomalize P if we're using smooth policy updates
        for(var i=0,n=poss.length;i<n;i++) {
          var a = poss[i];
          this.P[a*this.ns+s] /= psum;
        }
      }
    }
  }


  var DQNAgent = function(env, opt) {
    this.gamma = getopt(opt, 'gamma', 0.75); // future reward discount factor
    this.epsilon = getopt(opt, 'epsilon', 0.1); // for epsilon-greedy policy
    this.alpha = getopt(opt, 'alpha', 0.01); // value function learning rate

    this.experience_add_every = getopt(opt, 'experience_add_every', 25); // number of time steps before we add another experience to replay memory
    this.experience_size = getopt(opt, 'experience_size', 5000); // size of experience replay
    this.learning_steps_per_iteration = getopt(opt, 'learning_steps_per_iteration', 10);
    this.tderror_clamp = getopt(opt, 'tderror_clamp', 1.0);

    this.num_hidden_units =  getopt(opt, 'num_hidden_units', 100);

    this.env = env;
    this.reset();
  }
  DQNAgent.prototype = {
    reset: function() {
      this.nh = this.num_hidden_units; // number of hidden units
      this.ns = this.env.getNumStates();
      this.na = this.env.getMaxNumActions();

      // nets are hardcoded for now as key (str) -> Mat
      // not proud of this. better solution is to have a whole Net object
      // on top of Mats, but for now sticking with this
      this.net = {};
      this.net.W1 = new R.RandMat(this.nh, this.ns, 0, 0.01);
      this.net.b1 = new R.Mat(this.nh, 1, 0, 0.01);
      this.net.W2 = new R.RandMat(this.na, this.nh, 0, 0.01);
      this.net.b2 = new R.Mat(this.na, 1, 0, 0.01);

      this.exp = []; // experience
      this.expi = 0; // where to insert

      this.t = 0;

      this.r0 = null;
      this.s0 = null;
      this.s1 = null;
      this.a0 = null;
      this.a1 = null;

      this.tderror = 0; // for visualization only...
    },
    toJSON: function() {
      // save function
      var j = {};
      j.nh = this.nh;
      j.ns = this.ns;
      j.na = this.na;
      j.net = R.netToJSON(this.net);
      return j;
    },
    fromJSON: function(j) {
      // load function
      this.nh = j.nh;
      this.ns = j.ns;
      this.na = j.na;
      this.net = R.netFromJSON(j.net);
    },
    forwardQ: function(net, s, needs_backprop) {
      var G = new R.Graph(needs_backprop);
      var a1mat = G.add(G.mul(net.W1, s), net.b1);
      var h1mat = G.tanh(a1mat);
      var a2mat = G.add(G.mul(net.W2, h1mat), net.b2);
      this.lastG = G; // back this up. Kind of hacky isn't it
      return a2mat;
    },
    act: function(slist) {
      // convert to a Mat column vector
      var s = new R.Mat(this.ns, 1);
      s.setFrom(slist);

      // epsilon greedy policy
      if(Math.random() < this.epsilon) {
        var a = randi(0, this.na);
      } else {
        // greedy wrt Q function
        var amat = this.forwardQ(this.net, s, false);
        var a = R.maxi(amat.w); // returns index of argmax action
      }

      // shift state memory
      this.s0 = this.s1;
      this.a0 = this.a1;
      this.s1 = s;
      this.a1 = a;

      return a;
    },
    learn: function(r1) {
      // perform an update on Q function
      if(!(this.r0 == null) && this.alpha > 0) {

        // learn from this tuple to get a sense of how "surprising" it is to the agent
        var tderror = this.learnFromTuple(this.s0, this.a0, this.r0, this.s1, this.a1);
        this.tderror = tderror; // a measure of surprise

        // decide if we should keep this experience in the replay
        if(this.t % this.experience_add_every === 0) {
          this.exp[this.expi] = [this.s0, this.a0, this.r0, this.s1, this.a1];
          this.expi += 1;
          if(this.expi > this.experience_size) { this.expi = 0; } // roll over when we run out
        }
        this.t += 1;

        // sample some additional experience from replay memory and learn from it
        for(var k=0;k<this.learning_steps_per_iteration;k++) {
          var ri = randi(0, this.exp.length); // todo: priority sweeps?
          var e = this.exp[ri];
          this.learnFromTuple(e[0], e[1], e[2], e[3], e[4])
        }
      }
      this.r0 = r1; // store for next update
    },
    learnFromTuple: function(s0, a0, r0, s1, a1) {
      // want: Q(s,a) = r + gamma * max_a' Q(s',a')

      // compute the target Q value
      var tmat = this.forwardQ(this.net, s1, false);
      var qmax = r0 + this.gamma * tmat.w[R.maxi(tmat.w)];

      // now predict
      var pred = this.forwardQ(this.net, s0, true);

      var tderror = pred.w[a0] - qmax;
      var clamp = this.tderror_clamp;
      if(Math.abs(tderror) > clamp) {  // huber loss to robustify
        if(tderror > clamp) tderror = clamp;
        if(tderror < -clamp) tderror = -clamp;
      }
      pred.dw[a0] = tderror;
      this.lastG.backward(); // compute gradients on net params

      // update net
      R.updateNet(this.net, this.alpha);
      return tderror;
    }
  }

  // buggy implementation, doesnt work...
  var SimpleReinforceAgent = function(env, opt) {
    this.gamma = getopt(opt, 'gamma', 0.5); // future reward discount factor
    this.epsilon = getopt(opt, 'epsilon', 0.75); // for epsilon-greedy policy
    this.alpha = getopt(opt, 'alpha', 0.001); // actor net learning rate
    this.beta = getopt(opt, 'beta', 0.01); // baseline net learning rate
    this.env = env;
    this.reset();
  }
  SimpleReinforceAgent.prototype = {
    reset: function() {
      this.ns = this.env.getNumStates();
      this.na = this.env.getMaxNumActions();
      this.nh = 100; // number of hidden units
      this.nhb = 100; // and also in the baseline lstm

      this.actorNet = {};
      this.actorNet.W1 = new R.RandMat(this.nh, this.ns, 0, 0.01);
      this.actorNet.b1 = new R.Mat(this.nh, 1, 0, 0.01);
      this.actorNet.W2 = new R.RandMat(this.na, this.nh, 0, 0.1);
      this.actorNet.b2 = new R.Mat(this.na, 1, 0, 0.01);
      this.actorOutputs = [];
      this.actorGraphs = [];
      this.actorActions = []; // sampled ones

      this.rewardHistory = [];

      this.baselineNet = {};
      this.baselineNet.W1 = new R.RandMat(this.nhb, this.ns, 0, 0.01);
      this.baselineNet.b1 = new R.Mat(this.nhb, 1, 0, 0.01);
      this.baselineNet.W2 = new R.RandMat(this.na, this.nhb, 0, 0.01);
      this.baselineNet.b2 = new R.Mat(this.na, 1, 0, 0.01);
      this.baselineOutputs = [];
      this.baselineGraphs = [];

      this.t = 0;
    },
    forwardActor: function(s, needs_backprop) {
      var net = this.actorNet;
      var G = new R.Graph(needs_backprop);
      var a1mat = G.add(G.mul(net.W1, s), net.b1);
      var h1mat = G.tanh(a1mat);
      var a2mat = G.add(G.mul(net.W2, h1mat), net.b2);
      return {'a':a2mat, 'G':G}
    },
    forwardValue: function(s, needs_backprop) {
      var net = this.baselineNet;
      var G = new R.Graph(needs_backprop);
      var a1mat = G.add(G.mul(net.W1, s), net.b1);
      var h1mat = G.tanh(a1mat);
      var a2mat = G.add(G.mul(net.W2, h1mat), net.b2);
      return {'a':a2mat, 'G':G}
    },
    act: function(slist) {
      // convert to a Mat column vector
      var s = new R.Mat(this.ns, 1);
      s.setFrom(slist);

      // forward the actor to get action output
      var ans = this.forwardActor(s, true);
      var amat = ans.a;
      var ag = ans.G;
      this.actorOutputs.push(amat);
      this.actorGraphs.push(ag);

      // forward the baseline estimator
      var ans = this.forwardValue(s, true);
      var vmat = ans.a;
      var vg = ans.G;
      this.baselineOutputs.push(vmat);
      this.baselineGraphs.push(vg);

      // sample action from the stochastic gaussian policy
      var a = R.copyMat(amat);
      var gaussVar = 0.02;
      a.w[0] = R.randn(0, gaussVar);
      a.w[1] = R.randn(0, gaussVar);

      this.actorActions.push(a);

      // shift state memory
      this.s0 = this.s1;
      this.a0 = this.a1;
      this.s1 = s;
      this.a1 = a;

      return a;
    },
    learn: function(r1) {
      // perform an update on Q function
      this.rewardHistory.push(r1);
      var n = this.rewardHistory.length;
      var baselineMSE = 0.0;
      var nup = 100; // what chunk of experience to take
      var nuse = 80; // what chunk to update from
      if(n >= nup) {
        // lets learn and flush
        // first: compute the sample values at all points
        var vs = [];
        for(var t=0;t<nuse;t++) {
          var mul = 1;
          // compute the actual discounted reward for this time step
          var V = 0;
          for(var t2=t;t2<n;t2++) {
            V += mul * this.rewardHistory[t2];
            mul *= this.gamma;
            if(mul < 1e-5) { break; } // efficiency savings
          }
          // get the predicted baseline at this time step
          var b = this.baselineOutputs[t].w[0];
          for(var i=0;i<this.na;i++) {
            // [the action delta] * [the desirebility]
            var update = - (V - b) * (this.actorActions[t].w[i] - this.actorOutputs[t].w[i]);
            if(update > 0.1) { update = 0.1; }
            if(update < -0.1) { update = -0.1; }
            this.actorOutputs[t].dw[i] += update;
          }
          var update = - (V - b);
          if(update > 0.1) { update = 0.1; }
          if(update < 0.1) { update = -0.1; }
          this.baselineOutputs[t].dw[0] += update;
          baselineMSE += (V - b) * (V - b);
          vs.push(V);
        }
        baselineMSE /= nuse;
        // backprop all the things
        for(var t=0;t<nuse;t++) {
          this.actorGraphs[t].backward();
          this.baselineGraphs[t].backward();
        }
        R.updateNet(this.actorNet, this.alpha); // update actor network
        R.updateNet(this.baselineNet, this.beta); // update baseline network

        // flush
        this.actorOutputs = [];
        this.rewardHistory = [];
        this.actorActions = [];
        this.baselineOutputs = [];
        this.actorGraphs = [];
        this.baselineGraphs = [];

        this.tderror = baselineMSE;
      }
      this.t += 1;
      this.r0 = r1; // store for next update
    },
  }

  // buggy implementation as well, doesn't work
  var RecurrentReinforceAgent = function(env, opt) {
    this.gamma = getopt(opt, 'gamma', 0.5); // future reward discount factor
    this.epsilon = getopt(opt, 'epsilon', 0.1); // for epsilon-greedy policy
    this.alpha = getopt(opt, 'alpha', 0.001); // actor net learning rate
    this.beta = getopt(opt, 'beta', 0.01); // baseline net learning rate
    this.env = env;
    this.reset();
  }
  RecurrentReinforceAgent.prototype = {
    reset: function() {
      this.ns = this.env.getNumStates();
      this.na = this.env.getMaxNumActions();
      this.nh = 40; // number of hidden units
      this.nhb = 40; // and also in the baseline lstm

      this.actorLSTM = R.initLSTM(this.ns, [this.nh], this.na);
      this.actorG = new R.Graph();
      this.actorPrev = null;
      this.actorOutputs = [];
      this.rewardHistory = [];
      this.actorActions = [];

      this.baselineLSTM = R.initLSTM(this.ns, [this.nhb], 1);
      this.baselineG = new R.Graph();
      this.baselinePrev = null;
      this.baselineOutputs = [];

      this.t = 0;

      this.r0 = null;
      this.s0 = null;
      this.s1 = null;
      this.a0 = null;
      this.a1 = null;
    },
    act: function(slist) {
      // convert to a Mat column vector
      var s = new R.Mat(this.ns, 1);
      s.setFrom(slist);

      // forward the LSTM to get action distribution
      var actorNext = R.forwardLSTM(this.actorG, this.actorLSTM, [this.nh], s, this.actorPrev);
      this.actorPrev = actorNext;
      var amat = actorNext.o;
      this.actorOutputs.push(amat);

      // forward the baseline LSTM
      var baselineNext = R.forwardLSTM(this.baselineG, this.baselineLSTM, [this.nhb], s, this.baselinePrev);
      this.baselinePrev = baselineNext;
      this.baselineOutputs.push(baselineNext.o);

      // sample action from actor policy
      var gaussVar = 0.05;
      var a = R.copyMat(amat);
      for(var i=0,n=a.w.length;i<n;i++) {
        a.w[0] += R.randn(0, gaussVar);
        a.w[1] += R.randn(0, gaussVar);
      }
      this.actorActions.push(a);

      // shift state memory
      this.s0 = this.s1;
      this.a0 = this.a1;
      this.s1 = s;
      this.a1 = a;
      return a;
    },
    learn: function(r1) {
      // perform an update on Q function
      this.rewardHistory.push(r1);
      var n = this.rewardHistory.length;
      var baselineMSE = 0.0;
      var nup = 100; // what chunk of experience to take
      var nuse = 80; // what chunk to also update
      if(n >= nup) {
        // lets learn and flush
        // first: compute the sample values at all points
        var vs = [];
        for(var t=0;t<nuse;t++) {
          var mul = 1;
          var V = 0;
          for(var t2=t;t2<n;t2++) {
            V += mul * this.rewardHistory[t2];
            mul *= this.gamma;
            if(mul < 1e-5) { break; } // efficiency savings
          }
          var b = this.baselineOutputs[t].w[0];
          // todo: take out the constants etc.
          for(var i=0;i<this.na;i++) {
            // [the action delta] * [the desirebility]
            var update = - (V - b) * (this.actorActions[t].w[i] - this.actorOutputs[t].w[i]);
            if(update > 0.1) { update = 0.1; }
            if(update < -0.1) { update = -0.1; }
            this.actorOutputs[t].dw[i] += update;
          }
          var update = - (V - b);
          if(update > 0.1) { update = 0.1; }
          if(update < 0.1) { update = -0.1; }
          this.baselineOutputs[t].dw[0] += update;
          baselineMSE += (V-b)*(V-b);
          vs.push(V);
        }
        baselineMSE /= nuse;
        this.actorG.backward(); // update params! woohoo!
        this.baselineG.backward();
        R.updateNet(this.actorLSTM, this.alpha); // update actor network
        R.updateNet(this.baselineLSTM, this.beta); // update baseline network

        // flush
        this.actorG = new R.Graph();
        this.actorPrev = null;
        this.actorOutputs = [];
        this.rewardHistory = [];
        this.actorActions = [];

        this.baselineG = new R.Graph();
        this.baselinePrev = null;
        this.baselineOutputs = [];

        this.tderror = baselineMSE;
      }
      this.t += 1;
      this.r0 = r1; // store for next update
    },
  }

  // Currently buggy implementation, doesnt work
  var DeterministPG = function(env, opt) {
    this.gamma = getopt(opt, 'gamma', 0.5); // future reward discount factor
    this.epsilon = getopt(opt, 'epsilon', 0.5); // for epsilon-greedy policy
    this.alpha = getopt(opt, 'alpha', 0.001); // actor net learning rate
    this.beta = getopt(opt, 'beta', 0.01); // baseline net learning rate
    this.env = env;
    this.reset();
  }
  DeterministPG.prototype = {
    reset: function() {
      this.ns = this.env.getNumStates();
      this.na = this.env.getMaxNumActions();
      this.nh = 100; // number of hidden units

      // actor
      this.actorNet = {};
      this.actorNet.W1 = new R.RandMat(this.nh, this.ns, 0, 0.01);
      this.actorNet.b1 = new R.Mat(this.nh, 1, 0, 0.01);
      this.actorNet.W2 = new R.RandMat(this.na, this.ns, 0, 0.1);
      this.actorNet.b2 = new R.Mat(this.na, 1, 0, 0.01);
      this.ntheta = this.na*this.ns+this.na; // number of params in actor

      // critic
      this.criticw = new R.RandMat(1, this.ntheta, 0, 0.01); // row vector

      this.r0 = null;
      this.s0 = null;
      this.s1 = null;
      this.a0 = null;
      this.a1 = null;
      this.t = 0;
    },
    forwardActor: function(s, needs_backprop) {
      var net = this.actorNet;
      var G = new R.Graph(needs_backprop);
      var a1mat = G.add(G.mul(net.W1, s), net.b1);
      var h1mat = G.tanh(a1mat);
      var a2mat = G.add(G.mul(net.W2, h1mat), net.b2);
      return {'a':a2mat, 'G':G}
    },
    act: function(slist) {
      // convert to a Mat column vector
      var s = new R.Mat(this.ns, 1);
      s.setFrom(slist);

      // forward the actor to get action output
      var ans = this.forwardActor(s, false);
      var amat = ans.a;
      var ag = ans.G;

      // sample action from the stochastic gaussian policy
      var a = R.copyMat(amat);
      if(Math.random() < this.epsilon) {
        var gaussVar = 0.02;
        a.w[0] = R.randn(0, gaussVar);
        a.w[1] = R.randn(0, gaussVar);
      }
      var clamp = 0.25;
      if(a.w[0] > clamp) a.w[0] = clamp;
      if(a.w[0] < -clamp) a.w[0] = -clamp;
      if(a.w[1] > clamp) a.w[1] = clamp;
      if(a.w[1] < -clamp) a.w[1] = -clamp;

      // shift state memory
      this.s0 = this.s1;
      this.a0 = this.a1;
      this.s1 = s;
      this.a1 = a;

      return a;
    },
    utilJacobianAt: function(s) {
      var ujacobian = new R.Mat(this.ntheta, this.na);
      for(var a=0;a<this.na;a++) {
        R.netZeroGrads(this.actorNet);
        var ag = this.forwardActor(this.s0, true);
        ag.a.dw[a] = 1.0;
        ag.G.backward();
        var gflat = R.netFlattenGrads(this.actorNet);
        ujacobian.setColumn(gflat,a);
      }
      return ujacobian;
    },
    learn: function(r1) {
      // perform an update on Q function
      //this.rewardHistory.push(r1);
      if(!(this.r0 == null)) {
        var Gtmp = new R.Graph(false);
        // dpg update:
        // first compute the features psi:
        // the jacobian matrix of the actor for s
        var ujacobian0 = this.utilJacobianAt(this.s0);
        // now form the features \psi(s,a)
        var psi_sa0 = Gtmp.mul(ujacobian0, this.a0); // should be [this.ntheta x 1] "feature" vector
        var qw0 = Gtmp.mul(this.criticw, psi_sa0); // 1x1
        // now do the same thing because we need \psi(s_{t+1}, \mu\_\theta(s\_t{t+1}))
        var ujacobian1 = this.utilJacobianAt(this.s1);
        var ag = this.forwardActor(this.s1, false);
        var psi_sa1 = Gtmp.mul(ujacobian1, ag.a);
        var qw1 = Gtmp.mul(this.criticw, psi_sa1); // 1x1
        // get the td error finally
        var tderror = this.r0 + this.gamma * qw1.w[0] - qw0.w[0]; // lol
        if(tderror > 0.5) tderror = 0.5; // clamp
        if(tderror < -0.5) tderror = -0.5;
        this.tderror = tderror;

        // update actor policy with natural gradient
        var net = this.actorNet;
        var ix = 0;
        for(var p in net) {
          var mat = net[p];
          if(net.hasOwnProperty(p)){
            for(var i=0,n=mat.w.length;i<n;i++) {
              mat.w[i] += this.alpha * this.criticw.w[ix]; // natural gradient update
              ix+=1;
            }
          }
        }
        // update the critic parameters too
        for(var i=0;i<this.ntheta;i++) {
          var update = this.beta * tderror * psi_sa0.w[i];
          this.criticw.w[i] += update;
        }
      }
      this.r0 = r1; // store for next update
    },
  }

  // exports
  global.DPAgent = DPAgent;
  global.TDAgent = TDAgent;
  global.DQNAgent = DQNAgent;
  //global.SimpleReinforceAgent = SimpleReinforceAgent;
  //global.RecurrentReinforceAgent = RecurrentReinforceAgent;
  //global.DeterministPG = DeterministPG;
  })(RL);

    function makeBombMap(map, map_objects) {
        var bomb_map = [];

        for (var p in map_objects) {
            var bomb = map_objects[p];
            if (bomb.type === 'bomb') {
                bomb_map.push({x: bomb.x, y: bomb.y, birth: bomb.birth, value: 1});
                let radius = bomb.radius;
                // @TODO refactor

                for(let bx = 1; bx <= radius; bx++){
                    let new_x = bomb.x + bx;
                    if(map(new_x, bomb.y) === WALL) {
                        break;
                    }
                    bomb_map.push({x: new_x, y: bomb.y, birth: bomb.birth, value: 1 / (bx + 1)});
                }
                for(let bx = 1; bx <= radius; bx++){
                    let new_x = bomb.x - bx;
                    if(map(new_x, bomb.y) === WALL) {
                        break;
                    }
                    bomb_map.push({x: new_x, y: bomb.y, birth: bomb.birth, value: 1 / (bx + 1)});
                }
                for(let by = 1; by <= radius; by++){
                    let new_y = bomb.y + by;
                    if(map(bomb.x, new_y) === WALL) {
                        break;
                    }
                    bomb_map.push({x: bomb.x, y: new_y, birth: bomb.birth, value: 1 / (by + 1)});
                }
                for(let by = 1; by <= radius; by++){
                    let new_y = bomb.y - by;
                    if(map(bomb.x, new_y) === WALL) {
                        break;
                    }
                    bomb_map.push({x: bomb.x, y: new_y, birth: bomb.birth, value: 1 / (by + 1)});
                }
            }
        }
        bomb_map.forEach(function ({x, y, birth, value}) {
            bombMap[y][x] = {birth, value};
            var expireTime = BOMB_EXPLOSION_FINISH - Date.now() + birth;
            setTimeout(function () {
                if (bombMap[y][x].birth === birth) {
                    bombMap[y][x].value = 0;
                }
            }, expireTime);
        });
    }

    addBot({
        name: "derg",
        routine: dergachevBot
    });

    var bombMap = [];

    var inited = false;
    var agent;
    function init(size) {

        // create an environment object
        var env = {};
        env.getNumStates = function() { return size; };
        env.getMaxNumActions = function() { return 5; };

        // create the DQN agent
        var spec = {
            update: 'qlearn', // qlearn | sarsa
            gamma: 0.9, // discount factor, [0, 1)
            epsilon: 0, // initial epsilon for epsilon-greedy policy, [0, 1)
            alpha: 0.01, // value function learning rate
            experience_add_every: 30, // number of time steps before we add another experience to replay memory
            experience_size: 5000,  // size of experience replay memory
            learning_steps_per_iteration: 50,
            tderror_clamp: 1.0,
            num_hidden_units: 200
        };
        agent = new RL.DQNAgent(env, spec);
        var brain = localStorage.getItem('brain');
        if (brain) {
            agent.fromJSON({"nh":200,"ns":31,"na":5,"net":{"W1":{"n":200,"d":31,"w":{"0":0.024936659661233866,"1":-0.05937186426184301,"2":-0.029967935494636465,"3":0.02029597144312864,"4":0.004080719032451774,"5":0.026472330575487306,"6":-0.03249472994804169,"7":-0.007427828790439193,"8":-0.03343020492891718,"9":0.024904032341850138,"10":0.004919542066951048,"11":0.011530059921842653,"12":0.015790253619714697,"13":-0.010438778687181835,"14":0.030067120261985953,"15":-0.11498295647592818,"16":0.016552367907331344,"17":0.12177878788114488,"18":-0.10389873837861055,"19":0.018009026247504282,"20":0.37255400236986824,"21":-0.06411357828951927,"22":-0.025195742320074883,"23":0.02403755889567091,"24":0.20088755575322537,"25":-0.010779797551387641,"26":0.023988676545880216,"27":-0.0174645654066262,"28":-0.0050592131098391654,"29":-0.011205074251181871,"30":0.018075211289329117,"31":-0.04336991627791628,"32":0.06781573891150014,"33":0.027875115441587207,"34":-0.014614113892712709,"35":0.0993331981029199,"36":0.008968058362439479,"37":-0.07871746202907388,"38":-0.049374675590333074,"39":-0.023870078112585114,"40":-0.13996147472755807,"41":-0.04667277630082251,"42":0.05663650365667589,"43":0.030970217503759535,"44":-0.08111668748304274,"45":0.06994779626261688,"46":0.007662937091780896,"47":0.17509023298899526,"48":0.023634315640031992,"49":-0.060315260268344745,"50":-0.09078256436344696,"51":-0.049136415808330905,"52":-0.16648908006480798,"53":-0.02569776856027615,"54":-0.011765511873979933,"55":0.0876372486862488,"56":0.00790367928470898,"57":0.0018682065321246279,"58":0.04388247889093641,"59":0.007786132033868769,"60":0.0010088154003175332,"61":0.07545275181463953,"62":-0.02935925719073605,"63":0.06222321619394678,"64":0.03167646711426408,"65":0.0274698185584658,"66":0.006842354202963839,"67":-0.02449296682214991,"68":0.0005868218663163464,"69":-0.0632862732641795,"70":-0.04603734310634851,"71":0.01984591972029524,"72":0.011580036589553797,"73":0.046019023148010145,"74":0.07593008981729475,"75":0.038849814771770394,"76":-0.015390300440381546,"77":-0.15674281203191223,"78":0.023081416818852486,"79":-0.03703244689896771,"80":0.009721412170983209,"81":0.0372622957343537,"82":-0.0922812844660035,"83":0.0725830979095982,"84":0.08359079506435543,"85":-0.13546068997035524,"86":-0.3007812592834169,"87":0.02508229473699013,"88":0.004648454445043963,"89":0.037750245168438716,"90":0.015867535741493823,"91":-0.017455715526676418,"92":-0.02105326243598695,"93":-0.07417883534976204,"94":0.05871768894058297,"95":-0.010761123362964234,"96":-0.10771164337074242,"97":0.20039119290131122,"98":0.05557009699187678,"99":-0.01008140501033205,"100":-0.03368468449608921,"101":-0.0739112200870455,"102":-0.10322377991287808,"103":-0.0899328019410264,"104":-0.010650418373045673,"105":0.10624302563328708,"106":-0.08703904793125257,"107":0.18281951390985646,"108":0.06136975463565015,"109":0.00507918883700888,"110":-0.02768666883670271,"111":-0.10081013240791681,"112":-0.13907987780723258,"113":-0.09925582119107755,"114":0.025081807827800503,"115":0.00329591135057349,"116":-0.051306680314263824,"117":-0.19654132449588885,"118":0.09562036660976402,"119":0.07599054004645066,"120":-0.015530371247690277,"121":-0.016664107571741242,"122":-0.0466268353611162,"123":-0.06780776418835663,"124":-0.01074954809645717,"125":0.0030446596556336462,"126":-0.029060940441066205,"127":0.05135750211554656,"128":-0.0747456467544101,"129":0.018571484047027808,"130":-0.002775621550245808,"131":0.009996612622423497,"132":0.02288162042839296,"133":0.06161064324371925,"134":0.03872845596938971,"135":-0.018772336668112333,"136":-0.036913803318667004,"137":-0.0065981802213401905,"138":-0.027656645267613127,"139":-0.020979467778126402,"140":0.01595389919827566,"141":0.025474425676997335,"142":-0.018660376252853137,"143":-0.07431739789101992,"144":-0.008420044527054549,"145":0.03764585287388244,"146":0.0004052975172170308,"147":0.11085708419563398,"148":0.04277518040827981,"149":0.0050267695959122895,"150":0.029766946462233237,"151":-0.018553720309244192,"152":-0.026191301033126772,"153":-0.03415893201987179,"154":-0.03378927275770547,"155":0.027668214046564985,"156":0.034223641034457665,"157":0.0318839156130432,"158":0.018267602028634776,"159":0.08853811219567911,"160":-0.002814385675692452,"161":-0.011472887854591548,"162":0.008452351589081723,"163":0.018399212423443254,"164":-0.09255702318144703,"165":-0.0001295737812685105,"166":0.03195262416057834,"167":-0.017522544953627147,"168":0.0006104821251974197,"169":0.034853125734071184,"170":-0.014022013536522838,"171":0.016991221926098376,"172":0.02476345130388584,"173":-0.030151619420338643,"174":-0.050044922801081554,"175":0.05212383672566125,"176":0.01169973697461244,"177":0.03743079476168541,"178":-0.15354772788454193,"179":0.08256797631939698,"180":-0.021416674109006897,"181":-0.020470737550341525,"182":0.026105089136535028,"183":0.033720359562726676,"184":0.008696136822010106,"185":-0.021046283895569978,"186":0.01906132827927319,"187":0.03295834042788905,"188":-0.023868077370399393,"189":0.0748884543481976,"190":0.007192211828574045,"191":0.038573766376815075,"192":0.056635575962425536,"193":-0.07689658435862438,"194":-0.020397039688160146,"195":0.03861265358616866,"196":-0.02471618839113003,"197":0.032325156538251236,"198":-0.012870682927833757,"199":-0.09976195191040267,"200":0.04607972702300584,"201":-0.01436380553233387,"202":-0.03729909941293069,"203":-0.06617771660152791,"204":0.1398915459290842,"205":-0.014806269578275393,"206":0.08790187950738382,"207":0.05862174019078307,"208":0.09842228937346997,"209":-0.08851498752555957,"210":0.0017297391101675891,"211":-0.010223037061379758,"212":0.0003855404385565403,"213":-0.06129794008340492,"214":-0.05224733838617649,"215":-0.03355106921273985,"216":0.01694815333951276,"217":-0.009034550388842537,"218":0.028938205246639932,"219":0.02836855909439044,"220":-0.03497031028593865,"221":-0.03386794971993166,"222":0.011296717632371613,"223":-0.005456072834903843,"224":-0.018103933403932213,"225":-0.00658409379413412,"226":0.008994209686299355,"227":0.013167489592675876,"228":-0.037353924009955804,"229":-0.017327496749909053,"230":0.04468377933234532,"231":-0.013385055076723587,"232":0.054444407790896245,"233":-0.057394653015973136,"234":0.03296250197518961,"235":-0.045904467400788226,"236":-0.04851413044386234,"237":-0.021254037059117856,"238":-0.04867694549828862,"239":-0.032105925654266246,"240":0.10508237620933931,"241":0.17548453183590382,"242":0.013257413233317914,"243":-0.009355956860074541,"244":-0.01711152487344144,"245":-0.015267939354806366,"246":0.012698676472644577,"247":0.021867628131370772,"248":-0.026884992501820734,"249":0.027086198981121697,"250":-0.04636397392619116,"251":0.001853343695992746,"252":-0.058860189186566124,"253":0.022179733137694356,"254":-0.006982161836469761,"255":-0.03300350773706373,"256":-0.03883913042280557,"257":-0.11135881676648746,"258":-0.057776660500002575,"259":0.033697148750959874,"260":-0.024366064371691212,"261":-0.16049898787633554,"262":0.06106386521935946,"263":-0.008603661080214257,"264":-0.029257250743174035,"265":0.008165247427644053,"266":0.07607849280744558,"267":0.07459099738006325,"268":-0.07069737358133842,"269":0.016750783813360075,"270":-0.11989473512372495,"271":0.06163431851953667,"272":-0.02632587547549034,"273":0.027058292587290964,"274":0.038102859701954844,"275":-0.04163992374375135,"276":-0.03241578784406625,"277":-0.009401654094026286,"278":0.15622957985550912,"279":0.110459954539486,"280":-0.06443987181821773,"281":-0.025415868209173832,"282":-0.02205533172671217,"283":0.005844019326433453,"284":-0.049880186534320976,"285":0.0171976726935066,"286":0.02994287497415202,"287":0.0038302216193625486,"288":-0.0524423897063044,"289":-0.019804825251512795,"290":0.014139325450388552,"291":-0.03120128004769117,"292":-0.0918599666866577,"293":-0.004147224280333167,"294":0.08327538395718687,"295":-0.09419266553504022,"296":-0.05466000219948806,"297":0.0625529789230267,"298":0.016601753858496108,"299":-0.08984210703659434,"300":-0.00446520519899564,"301":-0.006398624433816899,"302":-0.14419208070387496,"303":0.12215636484573346,"304":-0.013927515137531338,"305":-0.01113045468583512,"306":0.013835134700335846,"307":-0.014676755366063981,"308":0.031187012339570948,"309":0.055182964482154345,"310":-0.04988649827639901,"311":0.007105153317335906,"312":-0.015369007055496068,"313":-0.07883653058703521,"314":0.04658850245889959,"315":0.006382136704131945,"316":-0.01413316713747443,"317":0.06042249769609387,"318":0.0268609974524014,"319":0.18439163410589565,"320":0.06719356660598404,"321":-0.06769334861474326,"322":0.06049544725740753,"323":0.1865575392420928,"324":-0.022848196981947934,"325":0.012287951631698777,"326":0.10200889753286602,"327":0.05108304672726838,"328":-0.1123329030723644,"329":0.0017578537125776963,"330":0.14277401356039077,"331":-0.06620128668449736,"332":0.07748458796362803,"333":-0.014842179667304618,"334":0.06397111103751751,"335":0.01861782855435769,"336":-0.04075296313787704,"337":0.023831347295445008,"338":0.03843446659242359,"339":-0.0015410052975891922,"340":-0.10923509930597361,"341":-0.07390501443243723,"342":0.10211769118487769,"343":-0.0016016207323083992,"344":-0.01006745610971275,"345":0.12146022785628478,"346":0.03151515067587671,"347":0.048348547678821505,"348":0.0075786064258066815,"349":0.01782793947930196,"350":-0.02841820148738243,"351":-0.008897329214250481,"352":-0.0043369364078469485,"353":-0.04521375370005704,"354":-0.0149063276946563,"355":0.08166861083181053,"356":0.06274698534130511,"357":0.0024504014931014738,"358":0.07778670286751294,"359":-0.07444683843503629,"360":-0.07948055642400671,"361":0.25286175960422147,"362":-0.06429705799892495,"363":0.0034191629911242047,"364":0.006110205694172281,"365":0.14935187602433592,"366":-0.009132246389091506,"367":-0.02461166485553945,"368":-0.008480779070527123,"369":-0.07805342554784024,"370":-0.015638145557433286,"371":-0.07982289581900816,"372":0.0762002253150344,"373":-0.04435798035628297,"374":-0.05874892783200677,"375":-0.08109644195680496,"376":0.16641892964715976,"377":0.01712289484848965,"378":-0.04845626515035335,"379":0.05562889440127098,"380":0.08798352092345732,"381":0.06315110610564738,"382":0.0018344492725874731,"383":-0.045890868911799956,"384":0.05541830714985278,"385":0.055466429352618044,"386":0.01914947469115611,"387":0.049207181301113775,"388":0.10111735555959608,"389":0.01194085760340584,"390":-0.033556333492934694,"391":0.06534672580519643,"392":0.31782993764651946,"393":-0.06722840973585884,"394":0.08212169504025353,"395":-0.08072094462539581,"396":0.25933729630467045,"397":0.01204059304300227,"398":-0.043803890453860746,"399":-0.0076093927387732765,"400":0.035917587461482346,"401":0.015662499677019923,"402":-0.08710718591597033,"403":-0.04030527299122969,"404":0.04471330712226919,"405":-0.014023120808695196,"406":-0.15863644334641158,"407":0.10595464841986194,"408":0.05919872614938884,"409":0.027739905947248258,"410":0.07862332632561828,"411":0.01735950970601693,"412":0.1062199385245983,"413":0.000051694950530173835,"414":0.025458671676228666,"415":-0.06272519748919263,"416":0.1485168193193818,"417":0.09822708359920432,"418":-0.018943977302655537,"419":0.07011924852736044,"420":0.1494260991602488,"421":-0.10416920938114066,"422":-0.08954381845296085,"423":0.20810514962501492,"424":-0.038139887835411015,"425":-0.06780493779663407,"426":0.05000769722312489,"427":-0.16689239507327505,"428":0.03483743060235143,"429":-0.03904878769203514,"430":0.01064106214656987,"431":0.020182136610612154,"432":-0.0102391656363219,"433":-0.16497857348534664,"434":-0.03661525805554537,"435":0.06682223097833687,"436":0.034401739740516864,"437":0.08961191623992745,"438":-0.08857699849261315,"439":-0.019774695893479556,"440":0.0005777064540482442,"441":-0.10093600277366147,"442":-0.06082401281929619,"443":-0.1574193256250892,"444":-0.028814120569814906,"445":0.037260643186199004,"446":0.01578980276001445,"447":-0.16297367902258744,"448":0.053023317609745005,"449":-0.06819657720715952,"450":-0.07441118075244246,"451":-0.08333595265488096,"452":0.05765327969396806,"453":-0.04377385202353987,"454":0.011639126637144054,"455":0.0198140231912935,"456":0.022970903611416637,"457":0.03676825672871849,"458":0.2557851487967892,"459":0.00018039399847735355,"460":0.0634100479923557,"461":-0.006304691607426725,"462":-0.011044385297159945,"463":-0.07387822257729525,"464":0.18978880182363653,"465":-0.01781181302168223,"466":0.06869735243105408,"467":0.054506871518934526,"468":-0.012979159843593319,"469":0.07179470774224032,"470":-0.05453835861038763,"471":0.006623143433316408,"472":-0.02971193240979861,"473":0.016299061313180218,"474":-0.051763289545374604,"475":-0.022369690126918387,"476":-0.012627744102304167,"477":0.003245095384571714,"478":0.027977099255120937,"479":0.06691072653818918,"480":0.026051944786346017,"481":-0.04088742962459403,"482":-0.03174858608622449,"483":0.007412244480786383,"484":-0.019764185490021447,"485":0.24014721592223578,"486":-0.10575264405888461,"487":-0.00019357452716576738,"488":0.10790294356511028,"489":-0.05524948748228548,"490":0.009302690953083562,"491":-0.0052239825622096515,"492":0.006480364567003016,"493":-0.011857612992490529,"494":0.0079292826073107,"495":-0.0010246494885373428,"496":-0.028255796836886858,"497":0.045097120203869905,"498":0.02004542411941725,"499":0.12706788114088077,"500":-0.025268337785263252,"501":0.024077789235143324,"502":0.014801423071275145,"503":-0.08016557245909302,"504":-0.10388552834592289,"505":-0.02831974369268775,"506":0.021931832121618172,"507":0.058296583029026405,"508":-0.06775413679368879,"509":-0.13203182538483738,"510":0.010735720856529254,"511":-0.0543073377473428,"512":-0.04090429063817074,"513":0.00377452103654927,"514":0.07049517921877677,"515":0.033083668672287514,"516":-0.09078918862092582,"517":0.03829939123567812,"518":-0.01762249239445643,"519":0.283806644773908,"520":0.1269770087693467,"521":0.007501690334534026,"522":0.05810386974025947,"523":-0.03914019435602938,"524":-0.06480933655763887,"525":-0.04366556699105789,"526":0.11489083799772,"527":-0.034376180814044276,"528":0.04376784529867806,"529":-0.005040744503669005,"530":0.021101004501553198,"531":0.07214447743214153,"532":-0.012726834335703156,"533":-0.026345384704803142,"534":-0.06834723853459643,"535":-0.00020170784894818792,"536":-0.15417703523696166,"537":-0.057126048912174855,"538":0.017545584460635234,"539":0.06640259469614218,"540":-0.10545096486448402,"541":0.07837449486403203,"542":0.007006655639297621,"543":0.01930105453116203,"544":-0.015510411994780864,"545":-0.03369913212964593,"546":-0.014118758118401825,"547":-0.08732711927848102,"548":-0.0508154550264715,"549":0.022170116181891658,"550":0.0836354206261732,"551":0.12498563001820112,"552":0.012183286135682519,"553":0.043201793080429894,"554":0.030702732999993668,"555":0.004785949864389934,"556":-0.015465219520502662,"557":0.07572922618664667,"558":0.009736041798212948,"559":0.05386105786728644,"560":-0.005900959266312618,"561":-0.010353647273615142,"562":0.02260023930448271,"563":0.01147953575952235,"564":0.03953122367331982,"565":0.04417026446616506,"566":0.07044448769229343,"567":0.07170349376508026,"568":0.034610354580237694,"569":0.010662888474238569,"570":-0.09695777681467427,"571":0.022137152266907274,"572":-0.0007285499326199775,"573":0.06012531799372433,"574":-0.030955540254272117,"575":0.04091811941067774,"576":0.0034210392469579015,"577":-0.06343311543472244,"578":-0.04048611807859905,"579":-0.0077940754109662525,"580":0.012314252956537156,"581":-0.017555716783251272,"582":-0.06807025995150533,"583":-0.012718036875409598,"584":-0.015519750556103985,"585":0.0024741874439620135,"586":-0.025518646207895408,"587":0.02645088253516652,"588":-0.11530946607626791,"589":-0.0841977909929292,"590":-0.1565001300812201,"591":0.21045416976284426,"592":-0.0493817332989525,"593":0.04848184818143701,"594":-0.21126543335051987,"595":-0.12681872947104292,"596":-0.023580811782902462,"597":0.09367378652059924,"598":-0.1009695345229603,"599":-0.032562905441758314,"600":-0.09743817371682492,"601":0.4019602093566787,"602":0.43523016102552403,"603":-0.005788350934306253,"604":-0.10833721894071399,"605":0.06461735176232933,"606":0.05996223045524238,"607":-0.13632356412168253,"608":-0.024707602831383884,"609":-0.19821162358458497,"610":-0.2848154300618571,"611":-0.11045922843594702,"612":-0.15853709868287713,"613":-0.3011811029938706,"614":-0.3391407605715295,"615":-0.03439858115378852,"616":0.38042277200541225,"617":0.05589201369581901,"618":0.07661842978400887,"619":-0.8861533047771375,"620":-0.0249385224251779,"621":0.005748760281668189,"622":-0.03638090614256863,"623":0.05701821913790122,"624":-0.04502420535186812,"625":0.03318411819623315,"626":0.058974191590356974,"627":-0.028904724917017656,"628":-0.05550949203804103,"629":0.016614600242655105,"630":-0.03964733608975992,"631":0.03214904505361636,"632":-0.04753648643292553,"633":-0.07182979759782564,"634":0.024005480651286843,"635":0.007001336491078997,"636":-0.04739550956984723,"637":0.023712504655903223,"638":0.032557241889067404,"639":-0.07286144193724718,"640":-0.018912028783677044,"641":0.08997545867029774,"642":-0.05034889444211093,"643":0.0433369858504085,"644":-0.000028615360860556317,"645":-0.02235354578236989,"646":0.01987250187288999,"647":-0.04078597400909273,"648":-0.04385866976273357,"649":-0.027147152340450023,"650":0.01613897583994414,"651":0.004840154553176443,"652":0.027681194229541198,"653":0.06831182181966491,"654":0.0028200267632237166,"655":0.04244034066496862,"656":-0.05783120555131755,"657":0.027532943338828435,"658":-0.061203039420089374,"659":-0.03612591302927512,"660":-0.03608153821165333,"661":-0.004977763906603601,"662":0.01623876650491622,"663":0.016604295060577933,"664":0.013788559627381626,"665":0.009145313811239224,"666":-0.007094191487469788,"667":-0.004361723361007816,"668":-0.07613620961285934,"669":0.0808839173546941,"670":-0.04523160990741937,"671":0.24856277813228014,"672":-0.04073333374194242,"673":0.03495233709365613,"674":0.05464433278086539,"675":-0.18411422753567047,"676":-0.011902800662433755,"677":-0.005574762742988565,"678":0.033370025921602486,"679":-0.0005182210903580741,"680":0.018164135517243475,"681":-0.02205955014482467,"682":-0.01769705518253361,"683":-0.04402215462094191,"684":-0.000904612979358123,"685":0.03584084343618252,"686":-0.0501943847842618,"687":-0.026243055487776723,"688":-0.05784465189413211,"689":0.04019630542729478,"690":0.02429479923855745,"691":-0.11284632547157043,"692":0.004221547195887728,"693":0.01771173740114823,"694":0.0026627721150687925,"695":-0.08935709333996845,"696":-0.01568502280631472,"697":-0.008324054516832144,"698":0.028911749861661708,"699":0.07981425592471893,"700":0.0055158264302445125,"701":0.12778704685944392,"702":-0.06652740399756955,"703":-0.03846945641668051,"704":-0.08900294321914401,"705":0.00895419574322672,"706":0.15718010508291128,"707":-0.04775798305451661,"708":0.00029890142565839575,"709":0.012192286697681767,"710":0.046544465985400775,"711":0.010250615463977129,"712":0.11953099174955381,"713":0.05511463412901096,"714":-0.05781037597968048,"715":0.04562201433121489,"716":-0.014200717205734762,"717":-0.12025699754675138,"718":-0.033317445730362154,"719":0.03176035159387274,"720":0.018420133566857434,"721":-0.06250160290965381,"722":0.09308108498930763,"723":0.044182256040391756,"724":-0.1292838428939236,"725":0.0865582158038489,"726":0.10505997390933489,"727":-0.11263231584480798,"728":0.1112099186493184,"729":-0.03136664939771143,"730":-0.031198056885676558,"731":-0.02788137071940923,"732":-0.05450570159318024,"733":-0.08655633409158624,"734":-0.004061481714761205,"735":0.0646663185397716,"736":-0.02088321814253229,"737":0.14807540622183832,"738":-0.02731180769237982,"739":-0.023405205297121204,"740":0.016364094630453522,"741":0.025710939375796427,"742":0.019051815302284877,"743":-0.026371151062345142,"744":-0.05478061652576771,"745":-0.07134040904026896,"746":0.09072624775011902,"747":0.22924643715533877,"748":-0.26959596056076074,"749":-0.12085679022055036,"750":-0.02342898310498488,"751":-0.20119871490355584,"752":-0.11086194295703965,"753":-0.23461428160172654,"754":0.196800946826385,"755":-0.3640697224153635,"756":-0.16890959093324337,"757":-0.2144323075737857,"758":-0.03681037029438131,"759":0.38847448403750795,"760":-0.7822395511454727,"761":-0.0622369258482885,"762":0.3206250914314955,"763":0.04956950464215091,"764":-0.6778002327084358,"765":-0.4098386129795375,"766":0.391764546593203,"767":0.3823271474194711,"768":0.9252699046499661,"769":0.1518154960625783,"770":0.7314549743889018,"771":0.13104226518034706,"772":-0.4379247750247386,"773":-0.39100728362990256,"774":1.1849651211660508,"775":0.8123455919091638,"776":-0.9364525531583686,"777":1.55360376243129,"778":1.2017236059161933,"779":0.014677472799584918,"780":2.1700674212938886,"781":0.15269568171305908,"782":0.38004210977629965,"783":-0.39711738371936794,"784":0.5558870758665155,"785":-4.858784142075844,"786":4.180603714361758,"787":2.5498689332111657,"788":3.254316653629358,"789":0.23123646157308456,"790":2.4319819137708043,"791":3.0678858714470683,"792":0.816212572351022,"793":0.18445266161746493,"794":-0.10720522345470215,"795":-3.2985222297666277,"796":-1.4336089374927319,"797":-4.2118455112122675,"798":-2.0286034285297476,"799":0.8641518135930034,"800":2.904125376404562,"801":1.7589439984747326,"802":2.137278113628517,"803":-0.2752707811342111,"804":0.7265408942768339,"805":-3.648950193142769,"806":0.04516176233984323,"807":-0.042665936899471874,"808":-0.05026834544944653,"809":-0.0366054462571239,"810":-0.08628732147296762,"811":0.02907327830463266,"812":-0.024933561678628396,"813":0.09527614356111382,"814":0.05256075672301273,"815":0.13982727429874908,"816":-0.010573877386870099,"817":-0.004168735825039661,"818":-0.06256100325759709,"819":0.04119567659620027,"820":-0.03252469762963554,"821":0.038248956445731536,"822":0.06835669498175456,"823":0.006373922437429322,"824":0.014255997124441677,"825":0.07687778856251182,"826":0.13779978428935344,"827":0.002512111969607501,"828":-0.017081655029656213,"829":-0.0166183481423811,"830":-0.09624168474829445,"831":0.0037236264806036166,"832":-0.02373956161856878,"833":-0.03046334488683476,"834":-0.010266787949006052,"835":-0.012949478864038874,"836":-0.018651608576261403,"837":0.11597626509643949,"838":0.19701360746644656,"839":0.03805266962828317,"840":0.15385375266403445,"841":0.162478050463347,"842":0.0006414424863573754,"843":0.33958821133747924,"844":0.14773141308270316,"845":-0.00041695874348825516,"846":1.0881381462765265,"847":0.2739845691350731,"848":-0.13794541619000622,"849":0.3816863044114032,"850":0.7826694265089563,"851":0.36552880324448966,"852":0.16791308851935738,"853":0.42182503424000184,"854":-0.26688150847303427,"855":0.5358093376944908,"856":1.6519629253061832,"857":-0.2077661138227386,"858":-0.769175562346374,"859":0.24545903409901776,"860":0.5818325384692832,"861":-0.4890764806162846,"862":-0.20728802111785943,"863":-0.40540448245728894,"864":0.3782040204160165,"865":0.6339176267015206,"866":1.7001615806088528,"867":-1.0439857540523874,"868":0.015955410845502255,"869":0.0036080074371478013,"870":0.011511262153489169,"871":-0.011104957398479103,"872":0.1581665007003827,"873":-0.0017186124679920904,"874":0.045180896090173915,"875":0.005777857615144991,"876":-0.0737201924292825,"877":-0.057937331501672025,"878":-0.03259659691551195,"879":0.016677513044663146,"880":0.00446188108015432,"881":-0.023887277000188785,"882":0.042610196506906256,"883":0.049315878658695,"884":-0.0352515387503446,"885":-0.01754752113210089,"886":0.0685542987754515,"887":0.01962027428342748,"888":0.15771417984534686,"889":0.07858520601014243,"890":0.010660816268950449,"891":-0.1421109010899547,"892":-0.16637024849039592,"893":-0.010563077444625056,"894":-0.020780879054770097,"895":0.004931193148874849,"896":-0.002569956529769355,"897":0.006646777728716102,"898":-0.03440027911031217,"899":0.03703103177013926,"900":0.05542597993353047,"901":0.09231579972141152,"902":-0.046145902720239745,"903":0.07178194705659284,"904":-0.05811573460870871,"905":-0.02728183530592988,"906":-0.04030867500135995,"907":-0.04806107430322563,"908":-0.0448016083267557,"909":-0.00904224918479663,"910":-0.006296111020061723,"911":0.06063929085566449,"912":0.08454501855609578,"913":-0.024918593016912106,"914":0.01281643137765756,"915":-0.024723980287636494,"916":-0.07028618046485625,"917":0.030889943474142215,"918":-0.1620823033812369,"919":0.194538123779567,"920":-0.050977859947949886,"921":0.09304476462790702,"922":-0.09059104785683167,"923":0.007472510406843833,"924":0.023371615282070487,"925":0.008003365587058671,"926":0.07475697630548042,"927":0.047574730077079644,"928":0.04297815076523747,"929":-0.038600815556147916,"930":0.008497827962293338,"931":-0.05718458186710626,"932":-0.01684984335594282,"933":-0.05360242768100316,"934":-0.17529908703985142,"935":-0.02367537596788472,"936":-0.024827624892598578,"937":0.0706294543875754,"938":-0.007871598280844678,"939":0.1791117672349373,"940":0.038247298179860646,"941":-0.10189322105342367,"942":0.044771110499860614,"943":0.061231511771366154,"944":-0.028199981320433803,"945":0.03062181099895477,"946":0.05819786435216772,"947":0.0052036476967325905,"948":0.042377197690301675,"949":0.0008877977860526902,"950":0.03462924609025059,"951":-0.019371827564347254,"952":-0.026931179723661113,"953":-0.0023896837435225414,"954":-0.13958021021802244,"955":0.01077452746203681,"956":-0.007483762655289802,"957":0.029206525222170043,"958":0.0334569682045682,"959":-0.014199929307587124,"960":-0.05598775908925592,"961":0.030584217161999834,"962":0.013740690374585692,"963":0.04684202686528927,"964":-0.001842656114797764,"965":-0.016058797923822803,"966":-0.025972839444300653,"967":-0.04392707314483407,"968":-0.028511681642356283,"969":-0.020453827124308627,"970":-0.1900562142493467,"971":-0.03496133169341724,"972":0.04085947279735334,"973":0.054603114385583885,"974":-0.09352204151225711,"975":0.06522065401429604,"976":-0.0712638978430908,"977":0.04082884799465425,"978":-0.03757833569505596,"979":0.06569908309287707,"980":-0.002973427020106701,"981":-0.084212624377036,"982":0.027174034146068393,"983":-0.05538865446384559,"984":-0.14620176182605743,"985":-0.25296824874700474,"986":-0.0033557913279682982,"987":-0.00471671983625331,"988":0.05552820294556987,"989":0.02112062788181881,"990":0.03524511383796855,"991":0.03466680917284717,"992":0.10943407118982681,"993":0.030097257322722146,"994":0.0037525366124415882,"995":0.13619185499377,"996":-0.023641551855678587,"997":0.07748208055084627,"998":0.010583547443349665,"999":-0.09351264512388972,"1000":0.028923637524344167,"1001":-0.059992391966135906,"1002":-0.014170951544994818,"1003":0.06980116308198646,"1004":-0.10272021349460997,"1005":-0.22847728238708315,"1006":0.043009257818155844,"1007":-0.059772354722903466,"1008":0.009909250046064187,"1009":0.08220097303628177,"1010":-0.06768223705501235,"1011":-0.022430632733766526,"1012":0.023763689690007403,"1013":0.07422092160856902,"1014":-0.01597739461388714,"1015":0.12723995437217017,"1016":0.45517447733466104,"1017":0.014682958425328742,"1018":0.06095181761762572,"1019":-0.08190804068194867,"1020":-0.00326953750135522,"1021":-0.014284070020198035,"1022":0.17249234449394815,"1023":0.06125849230968767,"1024":-0.04134885902815284,"1025":-0.0048306032469136746,"1026":-0.033848421980881456,"1027":-0.024636361179370108,"1028":-0.009268192873880305,"1029":-0.04811161595675464,"1030":0.051209451922929174,"1031":0.039740540197899105,"1032":-0.00759561390417009,"1033":0.002295907123287803,"1034":-0.03252186042255082,"1035":0.047177857735737475,"1036":0.04527344273033031,"1037":-0.04107787335147238,"1038":-0.001283828439018425,"1039":0.005494992228449144,"1040":0.025373795182521107,"1041":-0.04283067357634328,"1042":-0.06348048438498984,"1043":0.07631100619703987,"1044":-0.05123355097470218,"1045":0.007625791279512377,"1046":-0.1098059386942297,"1047":0.23828287346696142,"1048":-0.0014499477993257014,"1049":-0.026512072923740748,"1050":0.023251602892698467,"1051":0.0567158527852218,"1052":0.0343747296238266,"1053":-0.009254114059261211,"1054":0.010827514208058475,"1055":-0.047753390806146274,"1056":-0.03217128285470686,"1057":0.029275508166808243,"1058":-0.05720310278494244,"1059":0.02981945862397735,"1060":0.025928988546820266,"1061":0.0763583079311095,"1062":0.08040800860219885,"1063":0.025746332644981208,"1064":0.06499442600897157,"1065":-0.10091645559628286,"1066":0.012262046786116377,"1067":-0.008717764884209785,"1068":0.015906235515442402,"1069":0.06476028202307013,"1070":0.006301061196125091,"1071":0.06561200044583042,"1072":-0.09883205001874261,"1073":0.02900138663196796,"1074":0.1527248746209596,"1075":0.00011626245674199693,"1076":-0.054522834062899775,"1077":0.09475388315133691,"1078":0.31663459927990706,"1079":-0.008446151774255777,"1080":-0.004130195058367707,"1081":-0.022662211190620448,"1082":0.0041373159521979844,"1083":-0.012192371367266728,"1084":0.032447703171285286,"1085":0.020378888526807644,"1086":-0.021496072309227673,"1087":-0.08707560430074142,"1088":-0.007918467781861489,"1089":-0.041509959301676065,"1090":0.04925882925399812,"1091":-0.019358515770753804,"1092":0.018924554032528466,"1093":0.05423208240580321,"1094":0.01973428164407426,"1095":-0.011671811427243162,"1096":0.0011414857841388678,"1097":0.03477969266100456,"1098":-0.09826630793815211,"1099":0.0857699921729527,"1100":-0.04299300491270521,"1101":0.06464326577670623,"1102":0.0422497374594663,"1103":-0.05214161182472787,"1104":0.028178206343545634,"1105":-0.28068790925316006,"1106":0.070167024910229,"1107":-0.005486890102714864,"1108":-0.183492347216381,"1109":-0.05497925726956984,"1110":0.013453147761557883,"1111":0.010451605119624403,"1112":0.005037096944753211,"1113":0.015405247195258032,"1114":-0.033160496551758796,"1115":-0.0056842165930307475,"1116":0.006175634457043511,"1117":-0.052357383169895154,"1118":-0.058288805784092644,"1119":-0.023070900234904407,"1120":-0.025515246021847716,"1121":-0.03230550644108244,"1122":-0.04460160470756683,"1123":0.0756848012597914,"1124":0.055212772597938053,"1125":-0.07524672873136623,"1126":0.009271370670875563,"1127":-0.0364315782094434,"1128":0.004723307990855563,"1129":-0.019057744457653354,"1130":-0.021042535079512592,"1131":-0.0029125716784322657,"1132":0.0054335787660967395,"1133":0.013821191785822272,"1134":0.0188310224273488,"1135":0.0431491789441373,"1136":-0.18541180822691553,"1137":0.01443075444843721,"1138":-0.05290574181702222,"1139":-0.0747572002027645,"1140":-0.042871633668190495,"1141":0.01607583362121511,"1142":0.0201662102841046,"1143":0.03292140026905664,"1144":0.04522208117583209,"1145":0.024234697049030016,"1146":0.015025345860501976,"1147":-0.006185140200530689,"1148":0.031550857852815535,"1149":0.00731558794445429,"1150":-0.021764366003785583,"1151":0.11545191168445344,"1152":0.012366601473840744,"1153":-0.08210443098190862,"1154":-0.07830128681796024,"1155":-0.0523877126235817,"1156":0.04142319159916733,"1157":-0.01299802356095905,"1158":-0.018236171529153754,"1159":0.09032684093430583,"1160":0.048347406710013405,"1161":-0.02196707882615423,"1162":-0.0018065303870937789,"1163":0.003439429773814266,"1164":-0.12182097501707963,"1165":0.006268913140626931,"1166":0.15926218425428393,"1167":-0.10888549134274018,"1168":-0.07095498904646137,"1169":0.12712668213522144,"1170":0.06194979387320911,"1171":0.09450824163619384,"1172":0.049658828728247506,"1173":0.019969530126040358,"1174":-0.01380499574844174,"1175":0.017952697142465817,"1176":-0.026347924551749675,"1177":0.0880072848958776,"1178":0.1184026511034994,"1179":-0.049156389608302264,"1180":-0.00914647869780807,"1181":0.06896722655073666,"1182":-0.06473713136191483,"1183":-0.0018403296805223272,"1184":0.044053846162450255,"1185":-0.031811933492669066,"1186":-0.06758050304305913,"1187":-0.048790536578722496,"1188":-0.06564640406381073,"1189":0.0717521200172635,"1190":-0.09198182206083569,"1191":-0.10901025615830952,"1192":-0.050444062877719555,"1193":0.029182460118314813,"1194":-0.09688412191362081,"1195":-0.024504988517606963,"1196":0.07778072690113852,"1197":-0.027849529355281098,"1198":-0.06351635317392845,"1199":0.011659847995090178,"1200":-0.01862748107829846,"1201":-0.015890704851660425,"1202":0.29369155192674123,"1203":-0.036713739986800734,"1204":0.011118406133315675,"1205":-0.05245769903881251,"1206":-0.04724401315220397,"1207":-0.01913338994538715,"1208":0.15522916964917954,"1209":-0.016145609095233233,"1210":0.05693030628485643,"1211":0.01500750063167556,"1212":0.056551525756213986,"1213":0.17563569639067814,"1214":0.019930570801909582,"1215":0.05524920012243592,"1216":-0.010098500342450174,"1217":-0.0429793190957148,"1218":-0.16388444076034203,"1219":-0.03638856120753457,"1220":0.09572167308302364,"1221":-0.05063030263050392,"1222":-0.11101316532104472,"1223":0.051684575018390765,"1224":-0.048252153987430935,"1225":-0.06544702862324976,"1226":-0.0015053358347270483,"1227":-0.018320279114547918,"1228":-0.04707498131688106,"1229":0.186489060671111,"1230":0.11110861769815919,"1231":0.0488330315926421,"1232":-0.08354325244726409,"1233":0.10448744201798758,"1234":0.000317605744832525,"1235":0.020183194596155987,"1236":-0.006761564195207134,"1237":-0.05936402312381471,"1238":-0.025420060271171323,"1239":0.05016513340210723,"1240":-0.059037632729813674,"1241":-0.0036718283139081064,"1242":-0.02300639305560429,"1243":0.024288958045171526,"1244":-0.032271616446880945,"1245":0.06515327931085725,"1246":0.022162607907492606,"1247":-0.00732201428487038,"1248":0.01542676610638327,"1249":0.18851438181785146,"1250":0.021575194795893336,"1251":-0.03934722211772313,"1252":0.008524594800248007,"1253":0.014892013668455458,"1254":0.027685557032545573,"1255":0.046193143406455875,"1256":0.016451818008653765,"1257":0.008852457928562027,"1258":-0.0006638593966665694,"1259":0.06481583567340768,"1260":0.054468678814536126,"1261":0.025112800967013828,"1262":0.02979565386101263,"1263":0.13604964891867682,"1264":-0.06843499973949284,"1265":0.0027081150211729645,"1266":0.0052711851515625996,"1267":-0.040813503692996585,"1268":-0.041235458826851,"1269":-0.04313683626886414,"1270":-0.04605814899156556,"1271":0.05054967508650707,"1272":-0.01874774964515242,"1273":-0.016200278906977103,"1274":0.0522397685566912,"1275":0.02941026309336978,"1276":0.028754088721505325,"1277":0.05246341501271798,"1278":0.053610478979933804,"1279":0.06429673173253317,"1280":-0.024344893606919486,"1281":-0.008827345788101609,"1282":0.028815499139348864,"1283":-0.04221517505485009,"1284":-0.0300617414171761,"1285":-0.02969147263878571,"1286":0.019713331198113897,"1287":-0.021956461377170753,"1288":0.05442097551772609,"1289":-0.0670521663522071,"1290":0.027069567895473316,"1291":-0.17203307627188502,"1292":0.1103193976942837,"1293":0.0012346941662180755,"1294":-0.10319023729192299,"1295":0.13809278053962765,"1296":-0.026168260429081287,"1297":-0.030609851878574196,"1298":-0.02342799435650908,"1299":-0.003969941990348363,"1300":0.009313043933987493,"1301":-0.03392931279463022,"1302":-0.00421821253861794,"1303":-0.001249073645060796,"1304":0.004522177080366081,"1305":-0.08249826237638729,"1306":0.14286537864367532,"1307":-0.006292698422810849,"1308":-0.047528268584359726,"1309":0.0009431356483173404,"1310":-0.00557551816478519,"1311":0.062337131887626365,"1312":-0.017919181563227093,"1313":0.06413779325110419,"1314":-0.04617014525838031,"1315":0.08467088094110777,"1316":0.09095205574472671,"1317":-0.08461857333397131,"1318":0.1003654661353305,"1319":0.1669500401898164,"1320":-0.1081027979368582,"1321":-0.11196956301280733,"1322":0.41205863495432155,"1323":-0.22035116991070905,"1324":-0.023607259502548918,"1325":0.1302999896962521,"1326":0.1930272336959234,"1327":-0.0342971951689012,"1328":0.005377035914456916,"1329":0.03311811800253324,"1330":0.007711159874087646,"1331":-0.0220998618878937,"1332":-0.03566739107089224,"1333":0.004137250132638921,"1334":0.03449107735809863,"1335":0.06427436061908276,"1336":-0.06358791451576183,"1337":-0.03741623347732362,"1338":-0.033669816686901485,"1339":-0.015062621524415482,"1340":0.05693247587908362,"1341":0.02822461541084348,"1342":-0.003575656351504269,"1343":-0.005775695326939586,"1344":0.03661585141015865,"1345":-0.036406371436910015,"1346":0.12423562541455638,"1347":0.04430630426333809,"1348":-0.0610423464749365,"1349":0.02413252426345373,"1350":0.09365432636022837,"1351":-0.09197926336647638,"1352":-0.10925695125857207,"1353":0.058963976107835055,"1354":-0.07775298190976111,"1355":-0.016548855704191807,"1356":-0.11374941504063495,"1357":-0.11117082313934061,"1358":-0.036418797745722965,"1359":-0.04349473381877813,"1360":0.05505777998349448,"1361":0.05058540664808118,"1362":0.032654288397154285,"1363":-0.15147615525803462,"1364":0.00960861242713747,"1365":-0.0023000097825699854,"1366":0.023350978527926333,"1367":0.08942265378000779,"1368":-0.1748523813791864,"1369":-0.05113300590394426,"1370":-0.02895448300059078,"1371":-0.04072716751739939,"1372":-0.08368154009152134,"1373":-0.09778657091748032,"1374":-0.015822506372392617,"1375":0.03378793591950006,"1376":-0.05690856415388534,"1377":-0.12656749833471254,"1378":0.01368293039958149,"1379":-0.042685522539871554,"1380":-0.04152952224727559,"1381":-0.01969012832375427,"1382":0.15711891390720978,"1383":-0.012986266225053368,"1384":0.06995184529323313,"1385":-0.05473244271167637,"1386":-0.10842391304244428,"1387":0.14377277126260485,"1388":-0.1844905538947203,"1389":-0.04313552761660286,"1390":0.008277565092653469,"1391":-0.03220039489117926,"1392":-0.040077663706978496,"1393":0.03515847601697248,"1394":0.09611328434897545,"1395":-0.030520560814554357,"1396":-0.011520574919073235,"1397":-0.019862570477017585,"1398":0.03370349639096663,"1399":-0.08823060613023286,"1400":-0.025953297846193598,"1401":-0.008813341587747714,"1402":0.008180687494771854,"1403":0.04315811307771869,"1404":-0.002047138994098142,"1405":0.08155846291506455,"1406":-0.030732858116499544,"1407":0.08088374265444757,"1408":-0.00723179397833735,"1409":-0.0361182718730833,"1410":-0.05598878278819501,"1411":-0.0801100785008808,"1412":-0.04700656286304261,"1413":-0.0759184817768488,"1414":0.10501769678978588,"1415":-0.3000997753850246,"1416":0.07457547646333601,"1417":0.018929114239380852,"1418":-0.10329825087431675,"1419":0.10501412261007599,"1420":0.02366819672135641,"1421":0.018946226910162,"1422":0.016102039944882074,"1423":0.005861794970818218,"1424":-0.016569354861126713,"1425":0.07677208598238805,"1426":-0.0023152740893113426,"1427":-0.06136195724696106,"1428":-0.05018765173675794,"1429":0.036588772130475844,"1430":-0.003939272214477712,"1431":0.02656453283892793,"1432":-0.0498959481452983,"1433":0.029759140244991676,"1434":-0.01687137919321169,"1435":-0.14377464716685823,"1436":-0.02943326974958797,"1437":0.013074842488148908,"1438":-0.02869932858623106,"1439":-0.15013417717944805,"1440":-0.0030461302214565373,"1441":0.01740458121776028,"1442":-0.08623804331557713,"1443":-0.022258233338203138,"1444":0.03476365390526188,"1445":0.16461858927283607,"1446":0.00409215174888288,"1447":0.08267448917075977,"1448":-0.10688222831679699,"1449":0.056267615030704925,"1450":-0.0716408249111585,"1451":-0.0041949647906042945,"1452":0.039959077882032595,"1453":-0.0320283643131672,"1454":-0.05717539676171379,"1455":-0.02884687613991193,"1456":0.15108634338890692,"1457":0.01647039034802963,"1458":0.026336419075531196,"1459":0.041713316271656316,"1460":0.08903258961191438,"1461":-0.04059224155931808,"1462":-0.03494980146664809,"1463":0.022283226972316713,"1464":-0.05771948360280787,"1465":-0.027232460617435717,"1466":-0.12717491491415325,"1467":-0.006768512418975874,"1468":0.03227087062812186,"1469":-0.031188019179056354,"1470":-0.058623693752691386,"1471":-0.09054757426351787,"1472":0.024067819147966674,"1473":-0.14580667280864373,"1474":-0.09403677513431435,"1475":0.07602270390385697,"1476":0.006137313490509037,"1477":0.0055344255107227175,"1478":0.05231773370202181,"1479":0.008968659768251968,"1480":0.10343050430128693,"1481":0.21632735290299368,"1482":-0.019482813966580585,"1483":0.04038703852966631,"1484":0.00472288254770187,"1485":-0.053865799674048656,"1486":0.008625168125292593,"1487":0.15647864912866383,"1488":0.025374081684053734,"1489":-0.007245499334320822,"1490":0.029349668169553367,"1491":0.04849526499578761,"1492":-0.09581872755684215,"1493":-0.04072000433994241,"1494":-0.044008730320652056,"1495":-0.02861909448339682,"1496":-0.02138320358171654,"1497":0.03522482238962836,"1498":0.01011346710101924,"1499":0.01023810344907876,"1500":-0.0063166070809023405,"1501":-0.0004762329627864853,"1502":0.018302587171420116,"1503":-0.05502073519004237,"1504":-0.0034911502349471717,"1505":-0.012500368797451712,"1506":0.010400853711619477,"1507":-0.019009622339715346,"1508":0.05252582756163371,"1509":-0.11589134465840283,"1510":0.057113804885420866,"1511":0.061627040466062495,"1512":0.17941614840802947,"1513":-0.028669002875870684,"1514":0.0076717620701851575,"1515":-0.010171778883747455,"1516":0.017091534796609503,"1517":-0.013151045755790495,"1518":0.06582390255293376,"1519":0.003189177409756104,"1520":0.0008200865737181583,"1521":0.00022695006499016477,"1522":-0.12026583056409004,"1523":0.18811632295513694,"1524":0.023634248198979843,"1525":-0.05780697276278818,"1526":0.0599966130683339,"1527":0.0439532953847072,"1528":0.09657528854983055,"1529":0.06425786163569444,"1530":-0.048554206975886374,"1531":0.043139502401668514,"1532":0.16759977149709684,"1533":0.022962474115187974,"1534":0.03160032118214646,"1535":0.016817129850272172,"1536":0.01726267520669988,"1537":-0.17148287033417167,"1538":0.04998979800918078,"1539":-0.12434683853036714,"1540":-0.07601633795168528,"1541":0.1363054913365721,"1542":-0.12045272386572906,"1543":0.04194756651257565,"1544":0.043036148702179575,"1545":-0.011537817377946907,"1546":0.04179375318758262,"1547":0.053244179344192254,"1548":0.03136595255838113,"1549":-0.12282066137358501,"1550":-0.012192622496786019,"1551":0.002787791523354029,"1552":-0.05987903526674997,"1553":0.17934071328325607,"1554":-0.03588415861961691,"1555":0.056308912717648395,"1556":0.05542992526177206,"1557":-0.06321175693895052,"1558":-0.1307599297382681,"1559":-0.04836480605307909,"1560":-0.07473803897171494,"1561":0.0459335914175613,"1562":-0.012624417565577169,"1563":-0.18607631466919067,"1564":-0.011984781944038295,"1565":0.026910696794107767,"1566":-0.09314807639001478,"1567":-0.08802428932503766,"1568":0.13600459884703686,"1569":0.06394318903234554,"1570":0.07922759810875832,"1571":0.10966774944720152,"1572":-0.024018762146759416,"1573":0.14401235959391634,"1574":0.08395603689172038,"1575":-0.011370539597675143,"1576":0.04277330908232098,"1577":-0.11165667711827654,"1578":-0.11609167777217363,"1579":-0.08594922779582968,"1580":0.2656320709399306,"1581":-0.021891122516615815,"1582":-0.13619534679612735,"1583":-0.09230588454831858,"1584":-0.15336039623417552,"1585":-0.13184474602147805,"1586":-0.013944911644082066,"1587":-0.09937902061201621,"1588":0.04739679374641038,"1589":-0.026495411427306242,"1590":0.04413366968671709,"1591":0.012644012802517268,"1592":-0.06239731747896327,"1593":0.0281518886850397,"1594":0.006358918534095923,"1595":0.07044048396809667,"1596":-0.0055790525203013065,"1597":0.07479806072257603,"1598":0.05822793452141227,"1599":-0.0021800204708428065,"1600":-0.11860269826529296,"1601":-0.07540019941684879,"1602":0.060938440195990984,"1603":-0.11836280757546108,"1604":-0.06356291159597713,"1605":-0.2376554352104641,"1606":0.052034737012498604,"1607":-0.003682331450300893,"1608":0.0574849469161055,"1609":0.11843039572038902,"1610":-0.01605579471016773,"1611":-0.10896633353807746,"1612":0.001603718283929204,"1613":-0.014630135192321972,"1614":-0.0034462520751114036,"1615":0.0898454766697252,"1616":-0.09907832440625254,"1617":0.001444966647920607,"1618":-0.016185762275786747,"1619":-0.0010881876956157844,"1620":0.030910762605479557,"1621":-0.003733811510591551,"1622":0.02091722972740788,"1623":0.008532656341846606,"1624":-0.0072243250440928,"1625":-0.012005789306082043,"1626":-0.11972584155007282,"1627":-0.000931337450354958,"1628":-0.0353104482742131,"1629":-0.08772997656952893,"1630":0.06909642364548267,"1631":0.21674665617273184,"1632":0.00038359577130556394,"1633":-0.011494123366660692,"1634":0.030642316622574658,"1635":0.07104841811967848,"1636":0.11650542715135807,"1637":-0.00804954510212065,"1638":0.005832059324328762,"1639":-0.02936185941103352,"1640":0.01448992618417332,"1641":-0.026544574484055682,"1642":0.16992517903024554,"1643":-0.021235909986421657,"1644":0.0753502323630637,"1645":0.11602233867083331,"1646":-0.0712353408774418,"1647":0.07522105101331605,"1648":-0.08494824072861551,"1649":-0.007977122434803842,"1650":-0.06709750117551747,"1651":-0.013391436463019166,"1652":-0.11865760043918488,"1653":-0.040838195763657915,"1654":0.008820748976371804,"1655":0.11242065334907697,"1656":-0.04128603590711716,"1657":0.17538783644390515,"1658":-0.02860020391011477,"1659":0.07026500392683781,"1660":0.014615583671132451,"1661":-0.05613754617741664,"1662":-0.20387362303657472,"1663":-0.024152932608164604,"1664":-0.09668621670120885,"1665":-0.004274990836121916,"1666":0.002046983993435939,"1667":-0.09944737585002636,"1668":-0.033580185180267764,"1669":-0.0025472953614316175,"1670":0.09554017547698852,"1671":-0.005883656220676518,"1672":0.0474188503843464,"1673":-0.09423028051520875,"1674":0.05096776450556905,"1675":0.02312300450825429,"1676":0.06151572462763473,"1677":0.028224373849157474,"1678":-0.10111756177907961,"1679":0.01307364615089479,"1680":-0.009255387509355747,"1681":-0.06722152293488304,"1682":-0.05319934304111832,"1683":0.003652775639832538,"1684":-0.014294699722191863,"1685":0.02502472891238899,"1686":-0.02465629579680161,"1687":0.009897153049479709,"1688":0.054789915966392905,"1689":-0.06762206047305083,"1690":0.032921939080586854,"1691":0.07642115678120942,"1692":-0.036830535147194164,"1693":-0.17584644879333727,"1694":0.1305713120901401,"1695":-0.07376948741679458,"1696":0.03688680649222631,"1697":-0.019531954767462154,"1698":0.18394507558600257,"1699":-0.05256173982964053,"1700":0.010245465135160513,"1701":0.0013865130584474034,"1702":0.014150440514343528,"1703":-0.02335569863577035,"1704":-0.011750579542505074,"1705":-0.04058594697899168,"1706":0.09791697859077775,"1707":-0.06696255704596758,"1708":0.009942613656529363,"1709":-0.007290063928778414,"1710":-0.052814324934141836,"1711":-0.035692556737290174,"1712":-0.05385900182824395,"1713":-0.05581816297676175,"1714":-0.0828360901116555,"1715":0.0009157666867045603,"1716":-0.04869232660742738,"1717":0.016063787917367692,"1718":-0.11367521637550478,"1719":-0.01939257409544274,"1720":0.031361803931385865,"1721":-0.1374109178645259,"1722":-0.10587843634879986,"1723":0.035483468218024415,"1724":-0.0881261281149495,"1725":0.10857937796998468,"1726":-0.019747728096748667,"1727":0.06491465527816598,"1728":0.2713510139431218,"1729":0.43562924931163416,"1730":0.10053047127058735,"1731":0.21534765244837614,"1732":-0.02725171261878315,"1733":-0.09406833732033609,"1734":0.0037115114824551767,"1735":0.26921347842599236,"1736":0.052344704107313796,"1737":0.012370291333774508,"1738":0.01022180103359729,"1739":0.1690838637590128,"1740":-0.11367351011539356,"1741":-0.011568058133915848,"1742":0.027004619868393375,"1743":-0.06910764344966897,"1744":-0.07472325426316183,"1745":-0.1813272288629306,"1746":-0.021861336569958867,"1747":0.006728292435536623,"1748":-0.01618441810012312,"1749":-0.21040189958486075,"1750":0.029456871189017034,"1751":-0.026308378760081137,"1752":-0.034620958234048343,"1753":-0.05503732294315957,"1754":0.11433660792166037,"1755":-0.10683540883648039,"1756":0.0118785364140525,"1757":0.09939678185072828,"1758":0.021007384621446263,"1759":0.02839920741722047,"1760":0.2552173282577374,"1761":-0.03838515041976363,"1762":0.039304431664567714,"1763":-0.025181296562990885,"1764":-0.032552275737970496,"1765":-0.04984889778439884,"1766":0.15144972064001513,"1767":-0.008887033546713486,"1768":0.0690237932933147,"1769":0.07413283693422112,"1770":-0.07015742225526533,"1771":0.19772404280709382,"1772":-0.012622407218813674,"1773":0.038522504577009516,"1774":-0.0423450712609389,"1775":-0.03400793481198653,"1776":-0.09488659597282699,"1777":-0.06542963782375899,"1778":0.04986282697161976,"1779":0.01732528849659985,"1780":-0.00789221743016979,"1781":0.07725624313103173,"1782":0.04575083133411869,"1783":0.03689822116373798,"1784":0.042128790082415796,"1785":-0.05338622317171388,"1786":-0.13021254583215236,"1787":-0.07724776598802505,"1788":-0.0316707362328823,"1789":-0.041729505864693484,"1790":0.0071346696287364585,"1791":-0.00644295033940073,"1792":-0.013091553027986898,"1793":-0.026621584919076634,"1794":0.0461160658093767,"1795":0.017243482002989457,"1796":0.021489274642595195,"1797":-0.07866098714594398,"1798":0.017015418999553722,"1799":-0.03905250518165622,"1800":-0.01796031378244305,"1801":-0.11570619386078804,"1802":-0.06974321222449018,"1803":-0.03129902052141879,"1804":-0.0274320163861468,"1805":0.12621356482391555,"1806":0.06496193222095305,"1807":0.02072765997425852,"1808":-0.006732705956032677,"1809":-0.018731120626197993,"1810":-0.033007316931339785,"1811":0.04782660856300486,"1812":0.008388597329584897,"1813":0.04282778631370227,"1814":0.03385742896928046,"1815":0.10763625318329556,"1816":-0.07700321104948336,"1817":-0.05163329015400742,"1818":-0.12564423198796643,"1819":-0.05076403104516908,"1820":-0.11644384174043855,"1821":-0.18562103641430583,"1822":0.018453871649157284,"1823":-0.03012905628070777,"1824":-0.055262352345426716,"1825":0.05152956059772881,"1826":0.05420222067970728,"1827":0.04146719600442258,"1828":-0.09968524218096025,"1829":-0.09229699125945383,"1830":0.11594243613167371,"1831":0.052429270125878995,"1832":0.03018590117071266,"1833":0.03420725755612795,"1834":-0.019494834201605266,"1835":0.06287254872218934,"1836":-0.02701353035075647,"1837":0.020918566112548123,"1838":-0.1470089465023483,"1839":-0.0008851350184672142,"1840":0.045751623951241445,"1841":-0.029244818202114477,"1842":-0.03397276658149856,"1843":-0.02344989680603476,"1844":-0.024827353700178065,"1845":-0.06934008611342188,"1846":0.007336232131089499,"1847":-0.03038651735288172,"1848":-0.05782004095328253,"1849":-0.06407012893213908,"1850":0.047895781750967335,"1851":-0.034330218850024186,"1852":0.13197062716364244,"1853":-0.024346970693651496,"1854":0.009601113384526306,"1855":0.011666231184537836,"1856":0.03855013644149984,"1857":-0.025683009565463447,"1858":-0.008486845330023685,"1859":-0.01074366659072845,"1860":0.022193168843808123,"1861":-0.04883304730772719,"1862":0.0075119759385844035,"1863":-0.012018924078741893,"1864":-0.058330298584646036,"1865":0.025868631023674565,"1866":0.047158384942514765,"1867":-0.0029032276690816403,"1868":-0.029462581314952243,"1869":0.05541205765417015,"1870":-0.01696682556373573,"1871":-0.05110930592991298,"1872":0.011392527179429339,"1873":0.014398480279324343,"1874":0.0041246536685901735,"1875":0.028300269171398527,"1876":0.031543643072451735,"1877":0.05159111090981637,"1878":-0.02354824241160754,"1879":-0.14294186635866707,"1880":0.1842731326432651,"1881":-0.008732477421758495,"1882":-0.041522495660768814,"1883":0.02562291472972236,"1884":0.20545080291273546,"1885":-0.04195188178180855,"1886":-0.024563501000225852,"1887":-0.005512006778683152,"1888":0.00286092001789385,"1889":-0.012210426426359616,"1890":-0.04125087494299633,"1891":0.01484596788180435,"1892":0.0388732257690427,"1893":0.06897900395375124,"1894":0.04741709502088619,"1895":-0.021994337536881636,"1896":-0.006207679738094253,"1897":-0.0003271129943405163,"1898":-0.09518071627944315,"1899":-0.02468366801676369,"1900":-0.0528117556658149,"1901":0.02356486157325013,"1902":0.032998346945895235,"1903":-0.03769622474012534,"1904":-0.005665750859349894,"1905":-0.02696408951421324,"1906":-0.04619534187997117,"1907":-0.04322747759394687,"1908":-0.05778380345343888,"1909":0.011178352555241529,"1910":-0.050462868174902906,"1911":0.08036505123900589,"1912":-0.03491973229998183,"1913":0.07899046609816944,"1914":0.07655364602828986,"1915":0.18533646470600984,"1916":-0.02012245115938919,"1917":0.016726078128389547,"1918":-0.025644793115394984,"1919":-0.027682210860817267,"1920":-0.00723479151338498,"1921":0.060340682831377765,"1922":-0.10372270734656688,"1923":0.013651938661021592,"1924":-0.23586956257713038,"1925":-0.13273604550673104,"1926":0.03969184104697651,"1927":-0.09719994319656461,"1928":-0.011621687770848047,"1929":-0.35356561676246195,"1930":-0.26697347309275454,"1931":0.2103824614886235,"1932":-0.17681680179899628,"1933":-0.14888644248331295,"1934":-0.4170770748523847,"1935":-0.5447281871553075,"1936":0.15769188965704262,"1937":0.1304289203807183,"1938":0.006325676486071705,"1939":-0.3204404570763397,"1940":0.01958850988742757,"1941":-0.03106785129363983,"1942":0.31543943052692563,"1943":0.2739356850354996,"1944":-0.2868299290018009,"1945":0.48637594775584847,"1946":0.05435600475872483,"1947":-0.284108927189327,"1948":-0.42364897559406645,"1949":-0.7159308201966724,"1950":-0.04795430384766862,"1951":0.224504618112051,"1952":0.11838143551819416,"1953":-0.021825962274572605,"1954":-0.01453388624323789,"1955":0.014378782540181694,"1956":0.00850733454833211,"1957":-0.06786262785500512,"1958":-0.06998284485067703,"1959":-0.0314469590075094,"1960":-0.049722775819562776,"1961":-0.06187271205724055,"1962":-0.10163047629939727,"1963":-0.02749472145399332,"1964":0.07236603535332246,"1965":-0.035987345854542156,"1966":-0.12976397209259313,"1967":0.00218905879261078,"1968":-0.04968023936256057,"1969":-0.020049745384258476,"1970":0.007371207724976981,"1971":0.07654681077994491,"1972":-0.003422053401597143,"1973":-0.14980574110243633,"1974":-0.056149933682228016,"1975":-0.06983542879248124,"1976":0.10718238839570567,"1977":0.11601066197369699,"1978":0.011565432100298667,"1979":0.039385452234506575,"1980":0.0012742188340690713,"1981":0.012010105354037688,"1982":-0.001049000261653624,"1983":0.14003787236370976,"1984":0.018266969723064113,"1985":0.0072027082203150045,"1986":0.04655213074244721,"1987":-0.035110833104181796,"1988":-0.04842345496434189,"1989":-0.05333222353689242,"1990":-0.03433254745212935,"1991":0.002369148333139172,"1992":-0.001994557434134397,"1993":0.07737321737069991,"1994":0.019558269230424773,"1995":0.05532009846196797,"1996":-0.03576569846433091,"1997":0.068450472043842,"1998":-0.042023079853579344,"1999":-0.04505064834818536,"2000":0.01714409277070759,"2001":0.011139780739213113,"2002":-0.006826747496675324,"2003":-0.0038779018749331476,"2004":0.06124148426752968,"2005":-0.10382379841713754,"2006":0.038819124922337266,"2007":0.03290593081380713,"2008":0.13229927343061965,"2009":0.015412196339789954,"2010":0.014385454456782885,"2011":0.017013811711799068,"2012":0.04225849863717408,"2013":0.005822829847363296,"2014":0.03826836348331253,"2015":0.04268790320096546,"2016":0.01944973302274439,"2017":0.03978626663324866,"2018":0.06409198127939571,"2019":-0.07890636095915915,"2020":-0.011007723916618901,"2021":0.002843366874440764,"2022":-0.057134220342377376,"2023":-0.029840054725919496,"2024":0.022015019534708316,"2025":-0.02970151227596531,"2026":0.04423660594668334,"2027":-0.07493730076594625,"2028":0.014092546419387996,"2029":-0.04104640532500905,"2030":0.01994396972393435,"2031":-0.06468236236647898,"2032":-0.07483411244825793,"2033":0.07959589651520371,"2034":0.08862880353701834,"2035":-0.18908349566926314,"2036":-0.051549077170209075,"2037":0.11212178434764085,"2038":-0.015457112668015527,"2039":0.09161676817538211,"2040":-0.0189223178014686,"2041":0.007149052443083965,"2042":-0.05116695244544578,"2043":-0.010387782789880657,"2044":0.00872678619856121,"2045":0.08534621426371493,"2046":-0.006909506130827764,"2047":-0.04065563972265705,"2048":0.0025668092076234616,"2049":0.02268886180710523,"2050":0.0032969705271700222,"2051":-0.027188765451829897,"2052":-0.05699190339763316,"2053":0.006626329887819227,"2054":-0.02005386137015533,"2055":0.05154541409940567,"2056":0.028093081964284474,"2057":0.005146014145678637,"2058":0.006275078485622302,"2059":0.0834025581598962,"2060":-0.10931015371569718,"2061":-0.06856239316976481,"2062":-0.037371583335619865,"2063":-0.10436014668962748,"2064":0.0402067435551909,"2065":0.21297374349923198,"2066":0.03399560418902258,"2067":-0.04824834939351184,"2068":0.11469798430394364,"2069":-0.010206536060917015,"2070":-0.033804540469442694,"2071":-0.00389417713117701,"2072":0.002160463888324236,"2073":-0.03293452814265913,"2074":0.012175043890267656,"2075":-0.0015908766724301875,"2076":0.07202182128756057,"2077":0.013119962897133635,"2078":0.060641687979415604,"2079":0.03020563746594916,"2080":0.015331068334317115,"2081":0.041266875141365525,"2082":0.010444262479542842,"2083":-0.005762265893225506,"2084":0.01289248296966659,"2085":0.03774309213499016,"2086":0.06452129700053208,"2087":0.006034085223414187,"2088":0.01131064968640753,"2089":-0.03593872947030502,"2090":0.10434324313509767,"2091":0.06374113680438485,"2092":-0.032797649362910596,"2093":0.1488087186677977,"2094":0.01831630336048389,"2095":-0.015107783474615536,"2096":-0.1775584657146699,"2097":0.054598281033716375,"2098":-0.0234122402497681,"2099":0.10997779919917211,"2100":-0.13414801706587687,"2101":-0.05649914361909892,"2102":-0.01912104246373934,"2103":-0.030541737269530727,"2104":0.03555185453101776,"2105":0.043534910380242674,"2106":0.01535473006795372,"2107":-0.19946844598456223,"2108":0.017423014062903495,"2109":0.051118711932582515,"2110":0.07595457912252218,"2111":-0.07205877581573612,"2112":0.10764989752969575,"2113":-0.04602177612952742,"2114":0.049967250020220115,"2115":-0.03296012464813725,"2116":-0.03003904598931264,"2117":-0.06459595694298118,"2118":-0.05914400500608226,"2119":0.10160848851288933,"2120":-0.031034538403735323,"2121":0.009591145578694548,"2122":0.056617285522501636,"2123":-0.013639742188755138,"2124":0.00036202883434096104,"2125":-0.006717714411059653,"2126":0.051989363901122375,"2127":-0.06620905125809189,"2128":0.08324450747607925,"2129":-0.03404262721670937,"2130":0.03830967680005074,"2131":-0.13202384216839103,"2132":-0.2523279803729135,"2133":-0.050937427010344004,"2134":-0.03859488852554474,"2135":0.06310380419704639,"2136":0.015928580888526736,"2137":0.03276628007022515,"2138":-0.12288554075081987,"2139":-0.025238737762527323,"2140":0.010547320807944047,"2141":0.09332650916901354,"2142":0.029550471683544733,"2143":0.07701962406572614,"2144":-0.07637582279127313,"2145":0.006841598324389009,"2146":-0.040179140999457,"2147":-0.04473217114773038,"2148":-0.1015921681776785,"2149":0.049710736777654724,"2150":0.004060940982591124,"2151":0.04726337589662639,"2152":0.013548800496416782,"2153":-0.03665713208471879,"2154":-0.04047187647813403,"2155":-0.07980234072707465,"2156":-0.007577745405095438,"2157":0.022901087053181987,"2158":-0.022711815361701953,"2159":0.23902680378810628,"2160":-0.04116428822244109,"2161":0.03549013846036618,"2162":0.03733252114008715,"2163":-0.017886076816241405,"2164":-0.02159801054268942,"2165":-0.004497299684625803,"2166":0.0377235526481209,"2167":0.01830026186067474,"2168":0.038417469328596944,"2169":0.004014763301488102,"2170":-0.010526290891087907,"2171":-0.01913065385725448,"2172":-0.012325770550946741,"2173":-0.031183415730319526,"2174":0.01623392761242191,"2175":-0.0014024997512844203,"2176":-0.05277048535667163,"2177":0.03605452495651697,"2178":0.07798000360915508,"2179":-0.08551456796223167,"2180":0.02793753165215377,"2181":-0.004655473818137623,"2182":0.01160821842304963,"2183":-0.02663800169857515,"2184":0.01721886774010734,"2185":-0.038626421175676065,"2186":-0.0008450506031460805,"2187":-0.007781777702579572,"2188":-0.09916756421867581,"2189":-0.029311233841566364,"2190":-0.26364605179154976,"2191":-0.01914272713675098,"2192":0.026591370914542733,"2193":-0.05361231404343785,"2194":0.1871400079247902,"2195":0.024842547232227993,"2196":0.030833556531539286,"2197":0.050082996639269704,"2198":0.03714578119745582,"2199":0.0032212745260808756,"2200":0.033636696034397895,"2201":-0.01422025570126538,"2202":-0.049219462551913847,"2203":-0.023668887483334268,"2204":-0.03733345640735444,"2205":-0.008754783526967914,"2206":0.028778282179257507,"2207":-0.025272155185567958,"2208":0.0468299475740536,"2209":0.0002881069809710638,"2210":0.12577517407727512,"2211":0.034369466550750113,"2212":-0.04732613495303165,"2213":0.012982453594900379,"2214":0.10107972264244222,"2215":-0.08392703115948429,"2216":-0.012447722429703148,"2217":0.028527126747887606,"2218":-0.013825749748252616,"2219":-0.012818982894943536,"2220":0.05861098307183338,"2221":0.12477968655984197,"2222":0.009126337395554442,"2223":0.06733189426899104,"2224":-0.02954352005992214,"2225":-0.03696927759616666,"2226":-0.020027818216719097,"2227":-0.01845049088257393,"2228":-0.015516874494015977,"2229":-0.008823460148414376,"2230":-0.006083289362903565,"2231":-0.0710223494784735,"2232":-0.25566227559486737,"2233":0.020677542143681643,"2234":0.3129262403390975,"2235":0.3804321980887433,"2236":-0.04375424772353356,"2237":-0.17040138933781532,"2238":-0.00984901490501389,"2239":-0.1825598274784561,"2240":-0.369787501537398,"2241":0.4232822796214531,"2242":0.442506779564485,"2243":-0.14338407297138298,"2244":-0.007561161676311567,"2245":0.24847143000566377,"2246":-0.7395494929933026,"2247":-0.18308480759045045,"2248":-0.39968423461790276,"2249":-0.935624450476285,"2250":-1.8375399867074977,"2251":-1.5004925461200362,"2252":-0.6819824240146791,"2253":0.26063849599860267,"2254":-0.3523415896578825,"2255":-0.7058714831109385,"2256":-0.06670763689007679,"2257":0.4918000921872369,"2258":-0.7127235529455629,"2259":-0.7059331899958988,"2260":0.18020057697697972,"2261":1.0973944003970766,"2262":-0.28574972823721767,"2263":-0.021339866208261928,"2264":0.030177495347741057,"2265":0.014003424313461733,"2266":0.037917983119864294,"2267":-0.005995712321760656,"2268":-0.018115931601273713,"2269":-0.03287315598092128,"2270":-0.0015367389099698791,"2271":-0.011189698755815311,"2272":-0.0008404088057478893,"2273":0.005680251354911664,"2274":0.049501377471540695,"2275":-0.024100539175975662,"2276":0.002262226171692914,"2277":-0.018370293203176204,"2278":-0.02634035382383618,"2279":-0.015007492554636449,"2280":-0.012763503126218769,"2281":0.04868220947362017,"2282":0.07427849686340796,"2283":0.0331538714075275,"2284":0.010669646729231702,"2285":-0.037213812137193714,"2286":0.08920041762354214,"2287":-0.1532532407293111,"2288":-0.0010266737658849894,"2289":0.009505131472610907,"2290":-0.004452049173887939,"2291":-0.0049575402022242815,"2292":0.0031988749001426488,"2293":0.028770924711241607,"2294":0.07659128669030246,"2295":-0.05124372421249726,"2296":0.042078696289705876,"2297":0.021760311168452673,"2298":-0.15619486063101792,"2299":-0.08954013790372482,"2300":-0.04233906185789602,"2301":-0.039702897984709384,"2302":-0.0663608323496063,"2303":-0.06877408877100773,"2304":-0.023456486289586905,"2305":0.047057197712274755,"2306":-0.04954079521807063,"2307":-0.12790196629331885,"2308":-0.0274363131653142,"2309":0.03535984282022443,"2310":-0.0336304593714708,"2311":-0.08415619602892534,"2312":0.16508642996230313,"2313":-0.017487333118753817,"2314":-0.3155871094774786,"2315":-0.029408075742443748,"2316":-0.04686437596285619,"2317":0.024277308099400013,"2318":0.05314950831944054,"2319":-0.028390042146498238,"2320":0.023448666684560564,"2321":0.003468491988247544,"2322":0.038660167158566294,"2323":0.011188763156985884,"2324":0.1335177580107934,"2325":0.014511419450957559,"2326":-0.06333789653310253,"2327":-0.04880409635405951,"2328":-0.008419924156972793,"2329":0.045273287287524375,"2330":0.017298968197340087,"2331":-0.07577901032828685,"2332":0.08913594643153633,"2333":0.060243526348143936,"2334":0.035171727828652105,"2335":0.037917771517639726,"2336":-0.03292681970455468,"2337":0.014334606514187038,"2338":0.09944036912020784,"2339":0.008991512384115117,"2340":-0.013588009607039169,"2341":0.10924815173446717,"2342":0.05253003153692616,"2343":-0.06240459893884949,"2344":0.11041093135564128,"2345":0.03997001635905855,"2346":-0.03515915807710879,"2347":0.0633727337212575,"2348":-0.15963005489351728,"2349":0.05605873797318118,"2350":0.009275928158257176,"2351":0.003598014443845174,"2352":-0.0005259841923130171,"2353":0.07230810639820795,"2354":-0.007733081699145338,"2355":-0.020401794924787126,"2356":0.013294807815552243,"2357":0.02510722182079809,"2358":-0.016643099551094467,"2359":0.06199747584980579,"2360":0.0833507254503734,"2361":0.012669707307023124,"2362":0.004441579714943463,"2363":-0.02657674015440077,"2364":0.0068378505264089515,"2365":-0.13061768465958837,"2366":-0.04790396922111919,"2367":0.06445097249252471,"2368":0.005978433825474211,"2369":-0.14119970021501688,"2370":0.037435492698721824,"2371":-0.019132489011419243,"2372":0.016461747217167633,"2373":-0.04201190837819069,"2374":0.04484552916375364,"2375":0.07320811181250808,"2376":-0.18201761363611957,"2377":0.04443732309726793,"2378":0.007220791284292009,"2379":-0.09051806751673296,"2380":0.04824916992021853,"2381":-0.027201576011532846,"2382":0.007212072870675211,"2383":-0.012799070913678845,"2384":-0.009889643689512631,"2385":-0.010833378570102565,"2386":0.07240011932092784,"2387":0.005892003735577659,"2388":0.0027193674207838785,"2389":0.06165895916367375,"2390":-0.08841991909293859,"2391":0.11982038091536582,"2392":-0.06716547772199889,"2393":0.014206960437369095,"2394":0.04561774829677893,"2395":0.031211678922383364,"2396":-0.036137903834543426,"2397":0.031097113343060284,"2398":0.011130367191802415,"2399":0.0335549299191981,"2400":0.1459827922081561,"2401":-0.0693813978961606,"2402":-0.005129213545006164,"2403":-0.005081309001294941,"2404":0.012114521229786013,"2405":-0.06176836276944876,"2406":-0.04948010519009162,"2407":-0.14502928675325927,"2408":0.019981822442948903,"2409":0.06261824650999831,"2410":-0.1978921986198362,"2411":0.04277810580033269,"2412":0.012900635177554413,"2413":-0.029116598994756045,"2414":0.09738022179270472,"2415":0.06766426749693982,"2416":0.026920001714276207,"2417":-0.13313928774186168,"2418":0.015209784837660592,"2419":-0.0024086590861279564,"2420":-0.002244242774805493,"2421":-0.055978488865961344,"2422":0.10084333695513617,"2423":0.012205313768367353,"2424":0.00814740925942655,"2425":0.0417585379544523,"2426":-0.014280225728979624,"2427":0.053237404587150865,"2428":0.012782547154552607,"2429":-0.015808821167016675,"2430":-0.03244053281079724,"2431":0.13173194497713814,"2432":-0.06661318191527568,"2433":0.11238672737075615,"2434":-0.03029797442461147,"2435":0.007796502357630459,"2436":-0.0003274965541685827,"2437":0.07729837309809193,"2438":-0.019942598857843493,"2439":-0.020784676402554176,"2440":0.02029464531194129,"2441":-0.06341141795028997,"2442":-0.0548617884187296,"2443":0.005147328161950926,"2444":-0.0336017488169031,"2445":-0.013493836410636471,"2446":-0.0002694045760446179,"2447":0.0544170649492252,"2448":-0.05538449455594668,"2449":0.002734987230518677,"2450":-0.009177521542061438,"2451":-0.00848429601811831,"2452":-0.026279681288329436,"2453":-0.049457896167930365,"2454":0.041959111323831975,"2455":-0.016969675114740262,"2456":0.04252473695212642,"2457":0.024806322019972613,"2458":0.08629464451678968,"2459":0.04229835455734297,"2460":0.002652030318526524,"2461":-0.013763751632681565,"2462":0.11542924697493903,"2463":-0.039207404691963675,"2464":-0.03630604623984717,"2465":0.05198937035066411,"2466":0.05838390324615444,"2467":-0.06068426228709587,"2468":0.058702449701618184,"2469":0.09681217246506807,"2470":0.00922146214572932,"2471":0.04749473109799165,"2472":-0.14040982292750276,"2473":-0.11427905528253723,"2474":-0.023213105903731625,"2475":-0.04808703022236485,"2476":0.0023279008534073066,"2477":0.0144522877737204,"2478":0.004892387718202916,"2479":-0.09473574668151372,"2480":-0.0636322349130414,"2481":0.05342009094111619,"2482":0.036034351538464224,"2483":-0.04365065660860615,"2484":0.0856057337837714,"2485":0.0232707853069979,"2486":-0.007113154103139521,"2487":-0.009573697322821442,"2488":0.056370057224499,"2489":0.04030458167937659,"2490":0.07871537716593043,"2491":-0.04639374074511233,"2492":0.06968081382224356,"2493":0.12134615395732812,"2494":0.028891090899983523,"2495":-0.06688194308604468,"2496":0.08048539071165639,"2497":0.09428852927008322,"2498":-0.20704290487505234,"2499":-0.011961036558300742,"2500":0.22054241232161345,"2501":-0.08205256430086347,"2502":0.06007104172768185,"2503":0.06488984527220455,"2504":0.11028331901876565,"2505":-0.02035234959603768,"2506":-0.023758893703309723,"2507":0.04258001511045176,"2508":0.019498998893427013,"2509":-0.016703787770131137,"2510":-0.08893544603897949,"2511":0.00006594313498480217,"2512":-0.021430660334123455,"2513":-0.06148230523917913,"2514":0.001644385890491948,"2515":-0.039608239613225266,"2516":0.00899911689652574,"2517":-0.05475817490187752,"2518":0.030709829043052105,"2519":0.030060436133987085,"2520":-0.020692558321550304,"2521":-0.023429155180244774,"2522":-0.0214194191069235,"2523":-0.003438046774131502,"2524":-0.08596173046251467,"2525":0.030384988457658687,"2526":0.03350349736095569,"2527":-0.003750979292994554,"2528":-0.048578524476097125,"2529":0.014446838771496424,"2530":0.11320172213445293,"2531":0.07031098874417979,"2532":-0.05055339742991459,"2533":0.0021727511116962934,"2534":0.02898126723847508,"2535":0.15410742634916638,"2536":-0.008080852513623261,"2537":0.029015331624915743,"2538":-0.019710878587341232,"2539":-0.0170446430630128,"2540":-0.03213673553540126,"2541":0.11400317962897363,"2542":-0.007302072154331611,"2543":0.12557096576424825,"2544":0.08154674222813602,"2545":0.05881503133608956,"2546":0.020859889645269726,"2547":-0.05579959321041525,"2548":0.008782561908699853,"2549":-0.06265240632879406,"2550":0.04856234553735185,"2551":-0.11893665966735781,"2552":0.01996535573082565,"2553":-0.010316975147476492,"2554":0.05454345876332332,"2555":0.0629232470844351,"2556":-0.1350403858429972,"2557":-0.029709115732623907,"2558":-0.024177896459551318,"2559":-0.1055109223411068,"2560":-0.09825848517053926,"2561":-0.19486679459566203,"2562":-0.0007487355053788556,"2563":-0.005793107902715253,"2564":0.12021242784377867,"2565":0.20314468134850197,"2566":0.3968182032719546,"2567":-0.017447056349558766,"2568":0.08777360560591732,"2569":0.03565223920380546,"2570":0.010157850275771102,"2571":-0.018711432041944802,"2572":0.04576055948803092,"2573":0.059611696891432815,"2574":-0.02348954240504439,"2575":0.014927755512156204,"2576":0.029597228365004696,"2577":-0.04771035568586677,"2578":0.018491888757891338,"2579":-0.018128448250540972,"2580":-0.044122263907277415,"2581":-0.03054885281444885,"2582":-0.011873883529917393,"2583":-0.019653918799005365,"2584":0.059052358332590066,"2585":-0.03494665963234096,"2586":0.03343586943172735,"2587":-0.054029199258441074,"2588":-0.05128716101768055,"2589":-0.07750270837517342,"2590":-0.035111703751396585,"2591":0.0149680203693022,"2592":0.09229190547765866,"2593":0.08016598597913147,"2594":0.04169318458783231,"2595":0.07365248210374001,"2596":-0.15584038755585006,"2597":0.1572841098692269,"2598":-0.013029717317181961,"2599":-0.009720974865449946,"2600":-0.035593126847514873,"2601":-0.0037853061240320867,"2602":-0.014571182408992613,"2603":0.10982431878112367,"2604":-0.00523167844718778,"2605":-0.01309027191869533,"2606":-0.02103569683909936,"2607":0.010031996202021725,"2608":-0.13441235006567523,"2609":-0.024629419221306844,"2610":-0.04513912355732508,"2611":0.05671720354578191,"2612":0.033699762866095925,"2613":0.07213142181035714,"2614":0.04227319719291073,"2615":-0.03351832512400543,"2616":-0.031404518196140166,"2617":0.05286414794834002,"2618":-0.10917610537458391,"2619":0.024582583680974737,"2620":-0.00813185013173163,"2621":-0.027305415170615056,"2622":0.016957987159834834,"2623":0.12173532319303405,"2624":-0.06930012553088827,"2625":-0.05525052336046365,"2626":-0.028494350895355902,"2627":0.1375954310882529,"2628":0.10056222886823858,"2629":0.00786211684743566,"2630":0.003910582274674075,"2631":-0.024352893213934593,"2632":-0.007227456881911349,"2633":0.013324418425369813,"2634":0.0690713243396857,"2635":0.020255009529145183,"2636":0.014747808545288763,"2637":0.03142768889252829,"2638":-0.01912197826895591,"2639":0.006509143279953033,"2640":0.0059004354209558135,"2641":0.09045471113881919,"2642":-0.07831067424505915,"2643":-0.12622575254136734,"2644":0.05961325316492192,"2645":-0.0556027758599322,"2646":0.04252520115383162,"2647":-0.08687077661365268,"2648":-0.007964065599630947,"2649":-0.036603139761265675,"2650":0.05446637487832495,"2651":-0.11584639939638185,"2652":-0.019561877009124824,"2653":0.07409714370082192,"2654":-0.1422241339544572,"2655":-0.026886623382103396,"2656":0.048359976257447035,"2657":-0.023342347865742838,"2658":0.12094602561754339,"2659":-0.014918928630056942,"2660":-0.0033613276948525213,"2661":0.006959409421088215,"2662":-0.03914465803615345,"2663":-0.06291242050539965,"2664":-0.009092827090244037,"2665":0.0012502668916495724,"2666":-0.04351859161464561,"2667":0.015279008756756685,"2668":-0.04465492525093458,"2669":-0.007141882269601425,"2670":0.06250551519757852,"2671":0.019676812771417418,"2672":0.008794694672237724,"2673":0.04548120854689892,"2674":0.08135189623899318,"2675":-0.11635033525723207,"2676":0.00024011327292478845,"2677":0.022563450464006485,"2678":-0.01244321886990853,"2679":-0.10153334225621284,"2680":0.10503918492315768,"2681":-0.04552672483620718,"2682":0.06576917312631377,"2683":0.08974367278843444,"2684":-0.059024656422569786,"2685":-0.10619292281484982,"2686":0.13383082023617593,"2687":0.016080480251192884,"2688":-0.04943372747830231,"2689":-0.037373087908809076,"2690":-0.08533142484764447,"2691":-0.02415007858059495,"2692":-0.01423418022048631,"2693":0.03843132256279189,"2694":-0.0194170867459664,"2695":-0.0008069977995255234,"2696":-0.10831531680416585,"2697":-0.01467324626014699,"2698":-0.2040844325642818,"2699":-0.19061667846297448,"2700":-0.226371719320843,"2701":0.12284466415203495,"2702":-0.01477392805319459,"2703":-0.10433480408894708,"2704":0.021341618600501676,"2705":0.10151051231032181,"2706":-0.026201479455357973,"2707":-0.001328475287274011,"2708":-0.06042576313541442,"2709":0.0674400055380066,"2710":0.11100407301093086,"2711":0.1299664273443081,"2712":0.039984202808525375,"2713":0.14592944153622256,"2714":-0.015691948898498433,"2715":-0.16601422358163398,"2716":0.03524262805299454,"2717":0.17057432738117356,"2718":0.10530006587763224,"2719":-0.11553082219877175,"2720":-0.2326716533713841,"2721":-0.3133062431680906,"2722":0.04176981103803505,"2723":-0.10683601105145692,"2724":0.19995795388541562,"2725":0.3723591326486359,"2726":-0.04978932191688438,"2727":-0.26586854014564304,"2728":-0.013151347411813442,"2729":0.06440527220274522,"2730":0.009372141803915558,"2731":0.02775269685608383,"2732":0.1296047909704192,"2733":-0.01284488542283955,"2734":0.1135779536104036,"2735":-0.015724833260597284,"2736":-0.05216193040609131,"2737":-0.12325991571093807,"2738":-0.07096544988509271,"2739":0.12964286250473467,"2740":-0.08514093625849366,"2741":-0.12806307387136656,"2742":0.042103846158639026,"2743":0.0033529309337827122,"2744":-0.08544636096673675,"2745":0.07167513698718941,"2746":0.008380371567879496,"2747":-0.09654387958955545,"2748":-0.24985264271943614,"2749":0.128550103665757,"2750":-0.09308318293950762,"2751":-0.11450698860885455,"2752":0.026196199303798177,"2753":-0.00433340519694095,"2754":0.0029430480678058865,"2755":0.002951153166844712,"2756":-0.02584712711882499,"2757":0.0076699583862968575,"2758":0.007561469442789649,"2759":-0.060068868014801324,"2760":-0.009340834942267348,"2761":0.02814056168524495,"2762":-0.0625728581512752,"2763":0.10027800804037554,"2764":-0.010315971618554787,"2765":-0.06770504639131647,"2766":0.016226557728904526,"2767":0.10039381091729599,"2768":0.0850382031670346,"2769":0.1383765510936894,"2770":-0.15036378054708388,"2771":0.16631785447986233,"2772":0.11817661216847437,"2773":0.03055645485329837,"2774":-0.04447259547884806,"2775":0.06264132560050066,"2776":-0.01687160518299403,"2777":-0.17087530554149002,"2778":0.12463572823869862,"2779":0.10934106174232508,"2780":-0.09726834938667274,"2781":0.07402496206428552,"2782":0.1635114842514848,"2783":0.010600641517553192,"2784":0.029582138472880742,"2785":0.001500784628626107,"2786":0.06802764991376406,"2787":0.07631659386070173,"2788":-0.023353440051951176,"2789":-0.053097803560037994,"2790":-0.032055117315347416,"2791":-0.033879388522957805,"2792":-0.041521656412084884,"2793":-0.034859499992224784,"2794":-0.023011226858601477,"2795":-0.027660097190258963,"2796":-0.044582933307996515,"2797":0.10172063331419823,"2798":0.111481838901206,"2799":0.12024000299813609,"2800":0.1054119368103889,"2801":-0.1366433211577035,"2802":0.07515748009374407,"2803":0.08485442703780723,"2804":0.032713892151199554,"2805":0.03715913710806295,"2806":0.041403024691856395,"2807":0.0310635432014816,"2808":-0.09597941503378235,"2809":0.0530202208982357,"2810":0.1550817212302531,"2811":-0.08573508786337296,"2812":0.016149805422289002,"2813":0.051591035010874704,"2814":0.08212409641193316,"2815":0.006248432234685646,"2816":-0.015723385784138143,"2817":0.024924534510738977,"2818":0.02216344265316117,"2819":-0.00974617868724847,"2820":-0.04200484971337772,"2821":-0.04100863370295041,"2822":0.005311217064500263,"2823":-0.01403836923297002,"2824":-0.013806445659932925,"2825":0.06265071916952628,"2826":-0.0015300253031984223,"2827":-0.04803520398001636,"2828":-0.008015648104017442,"2829":0.016286780996255006,"2830":-0.06814091712992756,"2831":0.00540031996891322,"2832":0.0007302570638964382,"2833":0.04058476734607623,"2834":-0.05870053115377161,"2835":0.0650567590279695,"2836":-0.09216354508646629,"2837":0.07422885160795796,"2838":0.08661200806418591,"2839":-0.07912563692979949,"2840":-0.08606866859807477,"2841":-0.1505669953285197,"2842":0.022272768804084523,"2843":-0.05790209343756431,"2844":0.047448677606170435,"2845":-0.17893955784251414,"2846":0.02537830731363007,"2847":0.021287365463205594,"2848":0.028660778296848488,"2849":0.042867496273221256,"2850":-0.0022929696340741315,"2851":-0.061191418338913686,"2852":-0.030415400669035562,"2853":-0.01811264944327636,"2854":0.015706740396073744,"2855":-0.04824476129417922,"2856":-0.032016570069571475,"2857":-0.015587965203181798,"2858":-0.014902675850370261,"2859":0.025432184120553633,"2860":0.014381178467830168,"2861":0.062280580016348105,"2862":0.04989440504519236,"2863":-0.029035028727819615,"2864":0.04595317734044385,"2865":0.153748850045982,"2866":-0.13580454017963983,"2867":-0.020794655189894956,"2868":-0.007304894875367678,"2869":0.013928069045986431,"2870":-0.038116681317845444,"2871":0.09514002698995555,"2872":0.17991200812539432,"2873":-0.03416515987286173,"2874":-0.04322003950266929,"2875":0.14550979452481844,"2876":-0.14037676940026428,"2877":-0.004967507663083029,"2878":-0.01604782563324818,"2879":0.010396768587002562,"2880":-0.008152367501928605,"2881":0.03478207369954742,"2882":-0.03421664341171119,"2883":-0.05813796180042028,"2884":0.03198797307242883,"2885":0.03285095126277151,"2886":0.14426104340031662,"2887":-0.08208267280960321,"2888":-0.042084634496554155,"2889":-0.012065341120475047,"2890":-0.06596799347906927,"2891":-0.04446891063486738,"2892":-0.01577655953139859,"2893":0.030898385025396043,"2894":0.013736160035009017,"2895":0.05366213825622567,"2896":-0.056728676049582594,"2897":-0.026795343316897886,"2898":-0.0959927535172515,"2899":0.05542546582858596,"2900":-0.04477811405406069,"2901":0.020111391905712973,"2902":0.08477466561092911,"2903":0.07333112960373896,"2904":-0.05027884490445805,"2905":0.01856683241156207,"2906":0.25515677802100667,"2907":0.16632857466256148,"2908":-0.018658593761526888,"2909":0.035799068660569186,"2910":-0.02148645573907447,"2911":0.002167897722215977,"2912":-0.059644810097155526,"2913":0.13273925711969506,"2914":-0.054563857439358925,"2915":0.04730421983319476,"2916":0.034499167852944004,"2917":-0.035235203630482895,"2918":0.03147771452354379,"2919":-0.054454160144099616,"2920":-0.013276418610439003,"2921":-0.07673681629936459,"2922":0.04312992367999406,"2923":-0.06608639906940905,"2924":-0.021555241168427133,"2925":0.0025939516066266917,"2926":0.09347497871816624,"2927":-0.05562742025699478,"2928":0.07831506062172101,"2929":0.05742418727291055,"2930":0.08784602095061952,"2931":-0.061710534676629215,"2932":-0.0007682617181028194,"2933":-0.06103997933773528,"2934":0.18649897560387338,"2935":-0.16763079492128377,"2936":0.024002193450011797,"2937":0.13444340887378733,"2938":-0.006121526382711177,"2939":-0.007202163751243327,"2940":0.026546089273354122,"2941":0.06533153817258143,"2942":0.0009421809426769489,"2943":0.019437978945952165,"2944":0.022097450240977284,"2945":-0.022814351300146102,"2946":0.09420470522415862,"2947":-0.0034666797566935644,"2948":0.051006594622448845,"2949":0.06327298418217396,"2950":0.06119827646614238,"2951":0.052667474978427685,"2952":-0.09492676655169735,"2953":-0.05579718981313872,"2954":0.0015826851326641398,"2955":-0.05358559892615012,"2956":0.06880186799690786,"2957":0.010817745391553906,"2958":-0.10906056368807972,"2959":0.07227247802075651,"2960":-0.031355894370762354,"2961":-0.01703733975199557,"2962":-0.039460848995782144,"2963":0.0008608667145373811,"2964":-0.02814131960353956,"2965":-0.05096932352928291,"2966":0.07296376556842202,"2967":0.03286317298801634,"2968":0.09390728725635318,"2969":0.08020303218304432,"2970":0.04627949975275156,"2971":0.031158656294692864,"2972":-0.026321025737386528,"2973":-0.029025345038814408,"2974":-0.07121106136648396,"2975":0.08220447874885506,"2976":0.022923136772922147,"2977":-0.025608664281541314,"2978":0.0009689581417413025,"2979":-0.02209357959091203,"2980":0.009117344217725767,"2981":-0.020312739551632222,"2982":-0.001696187217927232,"2983":0.0283195829552409,"2984":0.026450612782352133,"2985":0.15865487311903023,"2986":0.05723434890246495,"2987":-0.08693083681094999,"2988":0.009728996691308864,"2989":0.05597633953948621,"2990":-0.00819636529498878,"2991":0.059933536369400756,"2992":0.01491138489597271,"2993":0.021416410176128293,"2994":-0.02104450744658668,"2995":0.024226353163298152,"2996":0.0622084461761312,"2997":-0.10408490264191306,"2998":0.030390340753859362,"2999":0.1773853612964419,"3000":0.18343432870145615,"3001":0.016133137926079906,"3002":-0.011589019502395142,"3003":-0.023865759160149075,"3004":0.005890658597579562,"3005":-0.020273231244454848,"3006":-0.005427879027844823,"3007":-0.03557136217874994,"3008":-0.05194504348560402,"3009":-0.06192846287155902,"3010":-0.04785045007867789,"3011":0.15370200990031932,"3012":0.028688085755491324,"3013":-0.03246975110289112,"3014":0.08329924354876651,"3015":0.04388133275380803,"3016":-0.025181754837200948,"3017":-0.011879277467922186,"3018":-0.02603942504264046,"3019":0.0375122416644214,"3020":0.00023993971034895728,"3021":0.044542089246954054,"3022":-0.009647916629445256,"3023":0.07599679205078343,"3024":0.042671059880355966,"3025":-0.06286178351545332,"3026":0.10016348735130863,"3027":0.13360149323428389,"3028":0.007121099940601378,"3029":0.016740079929653883,"3030":-0.1746369093605002,"3031":-0.06275346161491681,"3032":-0.014867693015641231,"3033":-0.028551468790544876,"3034":-0.0017333303693859196,"3035":0.009883155578907113,"3036":-0.011087780953013998,"3037":-0.06401798679232604,"3038":-0.06168737272150106,"3039":-0.00402090301348359,"3040":-0.02814259891995633,"3041":-0.04703488007984758,"3042":0.17497953661125568,"3043":0.04843508965882336,"3044":-0.048757317455105385,"3045":0.07371003447100966,"3046":0.10728055043608196,"3047":0.10451322400104689,"3048":0.08537056334467308,"3049":-0.11193453188137509,"3050":0.10961324302004156,"3051":0.1504505798964459,"3052":0.0012410774927445532,"3053":0.007094466401703178,"3054":0.13065770871931962,"3055":0.020706442753670923,"3056":-0.13559795440112693,"3057":0.017773852703802752,"3058":0.20626862845173388,"3059":-0.0017643743126741396,"3060":0.1121440632293832,"3061":-0.0978197820527551,"3062":-0.001290811770859555,"3063":-0.0037612662769315533,"3064":-0.0474183242537262,"3065":0.05407200843633779,"3066":0.06737392001277784,"3067":0.02244106014132596,"3068":-0.20678462632196923,"3069":-0.009647473754520094,"3070":0.037291971566324654,"3071":0.038099690411675616,"3072":0.03380209474623917,"3073":-0.0001175645398660248,"3074":-0.025408519336894784,"3075":-0.0132244093821969,"3076":-0.06414313164063468,"3077":-0.009170505037045083,"3078":0.021232165120081938,"3079":0.007891124167029655,"3080":0.026459745628875195,"3081":0.01832801884287488,"3082":0.008157255232364052,"3083":0.019937142291315928,"3084":-0.08317793018860421,"3085":0.04330324763292418,"3086":-0.030571676594636134,"3087":0.03338208619285348,"3088":0.034237898817167045,"3089":0.07829144086815404,"3090":-0.0912191910581721,"3091":0.0669714247212599,"3092":0.12442090363071419,"3093":-0.034989787572276894,"3094":0.01252605619752862,"3095":0.02403445113115695,"3096":0.011642070092556632,"3097":0.02891305385000294,"3098":-0.024384726084737827,"3099":0.0396799229790088,"3100":-0.008455566621656698,"3101":-0.048017832307104116,"3102":0.01294449584507642,"3103":-0.05075354781123245,"3104":-0.014163293396911771,"3105":-0.04146659783765486,"3106":-0.006186310229824924,"3107":0.06472222660538979,"3108":0.041961067101174164,"3109":0.0002221118073194598,"3110":0.023929606967268025,"3111":-0.029842964881935056,"3112":-0.0018917620147343413,"3113":0.07662952981698869,"3114":-0.09959241031586365,"3115":0.06902981829217121,"3116":-0.032310387595968854,"3117":-0.019132803262912834,"3118":0.0022953209685443086,"3119":0.07396511574242776,"3120":-0.23414636273196054,"3121":0.008886184864288322,"3122":-0.032314768919633606,"3123":-0.029452063844278318,"3124":-0.18733894415318225,"3125":-0.022057423205671358,"3126":0.006656956701250412,"3127":0.04247535610215762,"3128":0.02481413891209157,"3129":0.06270603378724116,"3130":-0.0746296675755726,"3131":-0.008245575625664508,"3132":0.029629182818090002,"3133":0.01341465070995436,"3134":-0.0016283799304042206,"3135":-0.0007981479861774029,"3136":-0.021291626634761786,"3137":-0.012787358042652358,"3138":-0.053306911661100645,"3139":-0.03254217024203707,"3140":-0.13378513088001456,"3141":-0.013859547960096346,"3142":0.032746043992313185,"3143":-0.06110923462340313,"3144":-0.10621015573678382,"3145":0.04237701331530907,"3146":0.03348555541107547,"3147":-0.05200248660619012,"3148":0.029421924475680917,"3149":0.0017007271212056113,"3150":0.05079653220754755,"3151":0.15057006914299087,"3152":-0.043930633789189745,"3153":-0.15111737831672492,"3154":0.22586244093300467,"3155":0.04098245265612306,"3156":0.0337265466126808,"3157":0.030787933141822262,"3158":-0.02152037630764026,"3159":-0.039833373591562304,"3160":0.0008730405888999569,"3161":0.15846244159452763,"3162":-0.020750200280006268,"3163":0.01071077814701849,"3164":0.01186573289034017,"3165":-0.07041283300814104,"3166":0.060560932276061984,"3167":-0.03896842073237852,"3168":-0.016222716509834423,"3169":0.028117745345793803,"3170":0.04872010422492541,"3171":-0.007566308362455582,"3172":0.015341534848991027,"3173":0.008623419427554804,"3174":-0.026031040536238063,"3175":0.00633333705603034,"3176":0.005455668273213687,"3177":0.05031745082463121,"3178":-0.022005876478760105,"3179":-0.00747443564617804,"3180":0.011027017645146308,"3181":0.2410005017938685,"3182":0.038620804957828714,"3183":-0.13416499618994707,"3184":-0.05025233587947965,"3185":0.17175631779536357,"3186":-0.05331639570243513,"3187":0.04467040120702876,"3188":-0.0023742271368783533,"3189":0.014395650833334288,"3190":0.0068321799072598105,"3191":0.020751911196000002,"3192":0.06948898050519561,"3193":-0.05666049836280609,"3194":-0.028041810175469528,"3195":-0.4708969251498685,"3196":0.10405263193944807,"3197":-0.11387599513901561,"3198":0.08421253187837968,"3199":-0.1656505533412347,"3200":-0.19269832471252768,"3201":0.0433838708951811,"3202":-0.46380555103049087,"3203":-0.12260802166038187,"3204":-0.2698303742507074,"3205":-0.4486890306864805,"3206":-0.08175126863179459,"3207":-0.5106029052765478,"3208":0.12239104367551726,"3209":-0.7143505041095658,"3210":-0.5948220026123178,"3211":0.1976476650408338,"3212":1.0314602959313623,"3213":-0.3784699823064711,"3214":-0.6235197777299151,"3215":-0.04755340034303622,"3216":1.261448391915435,"3217":0.20665252819307772,"3218":-0.14916710026957328,"3219":-0.25202108596899764,"3220":-1.1929834711219907,"3221":-0.07151312546386929,"3222":0.07619694174155094,"3223":-0.056429272979972335,"3224":-0.007921151252110936,"3225":-0.008839923307098402,"3226":-0.04322532077776503,"3227":-0.006999748605914917,"3228":0.004402532634298586,"3229":0.04249378771944383,"3230":-0.06998513183224848,"3231":0.017515699404614074,"3232":-0.019075339328303707,"3233":-0.11144928023642069,"3234":-0.0301300177563261,"3235":0.010176897673159699,"3236":-0.014747015587546685,"3237":-0.0068347800130752,"3238":0.011998790940754064,"3239":-0.035363524865622574,"3240":0.017718512516299526,"3241":0.08138509937786026,"3242":-0.06227481420600061,"3243":0.010786418585331423,"3244":0.20325108630161623,"3245":-0.020021500620126177,"3246":-0.05891380893260131,"3247":-0.08304153630291086,"3248":0.037119036621648276,"3249":-0.037660103680721,"3250":-0.008396201576204554,"3251":-0.017394567237925665,"3252":-0.025124707144432424,"3253":0.020108144620135424,"3254":0.01675825668621032,"3255":-0.04723031190408015,"3256":2.3365745717579975,"3257":2.687607040190183,"3258":0.27527727567950167,"3259":1.28141558056215,"3260":-0.12356999000328213,"3261":1.6743655270288402,"3262":-1.9736856669332659,"3263":-0.7119472662844539,"3264":-4.037892886473914,"3265":-0.8892151564332903,"3266":1.639181504534132,"3267":-1.1692817779100113,"3268":-3.8478799434305953,"3269":-3.746558172998913,"3270":1.1308947276286183,"3271":-0.09246236932408497,"3272":-3.41233113737311,"3273":-0.3660121849738371,"3274":-0.7035793702373734,"3275":1.6706356086839236,"3276":-2.299595463070082,"3277":-3.5261583200289475,"3278":2.0073337938187406,"3279":1.6903962036231188,"3280":2.180130631092864,"3281":1.7659927580413084,"3282":-7.006591034690366,"3283":6.140781739046712,"3284":-9.231452355552392,"3285":4.723776154224648,"3286":-0.025892478011655385,"3287":-0.03828463032024532,"3288":0.11928143238361884,"3289":-0.10394809100958483,"3290":-0.009200034884737357,"3291":-0.04748313036484052,"3292":-0.08620750501376355,"3293":0.10409840065687005,"3294":0.12674927132936847,"3295":0.05222163314404567,"3296":0.10489148724999273,"3297":-0.11083924386697229,"3298":0.15260317155600672,"3299":0.2784983262046186,"3300":-0.019286276802933974,"3301":-0.014186983612999029,"3302":0.07254080665437514,"3303":-0.05906599480416217,"3304":-0.034466337779601994,"3305":0.08880496142071935,"3306":0.033019286494621415,"3307":-0.01789890349522004,"3308":0.05771680686888834,"3309":-0.18233353745146771,"3310":-0.2316127550893118,"3311":0.04226583685800773,"3312":-0.03672967006932518,"3313":0.1558793113819106,"3314":0.11470792255255459,"3315":-0.049825645464661436,"3316":-0.29174938059835054,"3317":0.035445215825501086,"3318":0.02674209806760929,"3319":0.08089647186638585,"3320":0.037884296813102965,"3321":0.1627919011125842,"3322":-0.008222115101825762,"3323":0.008177823173248257,"3324":-0.043172178388098066,"3325":0.08390455227014067,"3326":-0.10036729557141141,"3327":0.014121206479655545,"3328":-0.0015935488079695656,"3329":0.08499063395044257,"3330":0.0766325855910146,"3331":-0.012547036932280618,"3332":0.003344188327537107,"3333":0.031051393637398658,"3334":-0.04423795055782399,"3335":-0.08797333621514294,"3336":-0.020054888251821668,"3337":0.11011994686003755,"3338":0.0173273966200366,"3339":0.14073377200206016,"3340":-0.2070159582312472,"3341":-0.026329971872459168,"3342":-0.044928351328181265,"3343":-0.045149677696167116,"3344":0.05781884360846488,"3345":0.046341185542287035,"3346":0.04043139816095184,"3347":-0.12419573448718497,"3348":-0.04005756922730793,"3349":0.03925101629974559,"3350":-0.0666473220397066,"3351":0.128103489081657,"3352":-0.07907719918584624,"3353":0.06931579091595291,"3354":0.021400471609715935,"3355":-0.021507833967593877,"3356":-0.04651528119532637,"3357":0.09954558865779331,"3358":0.00031210984640697313,"3359":-0.025063659688834703,"3360":-0.0787567149167497,"3361":-0.07726149971309296,"3362":0.03967541395290097,"3363":0.023640754868357622,"3364":0.07177585913795932,"3365":-0.06353989134950178,"3366":0.09536741776918146,"3367":0.05250196277903788,"3368":0.29653162225698415,"3369":-0.024237406710104818,"3370":0.024493542317386954,"3371":0.2595365656387808,"3372":0.08741015123150798,"3373":-0.016237244925942113,"3374":-0.00270685514877885,"3375":-0.1516338473775153,"3376":-0.028154372163235387,"3377":-0.06203581345484874,"3378":0.11874472581279058,"3379":-0.06660289394513852,"3380":-0.0036712976381282174,"3381":-0.008228024732060761,"3382":-0.09009039078016105,"3383":-0.015635168726087282,"3384":-0.02534325511634555,"3385":0.016738435990972456,"3386":0.08566402381683355,"3387":0.01862795888131712,"3388":0.11877754129948748,"3389":0.026689963994409285,"3390":-0.06623879136230654,"3391":0.061031766486932766,"3392":0.1130900352786262,"3393":-0.07525053210555589,"3394":0.046246142385029984,"3395":-0.014513860370167971,"3396":-0.0304760247523529,"3397":0.013735113341054095,"3398":0.018745419689113443,"3399":-0.12100569608658973,"3400":0.05747774397458982,"3401":-0.030809858761247023,"3402":-0.051401423275368834,"3403":-0.2341963850141148,"3404":0.04686941623493101,"3405":-0.01852717673344918,"3406":0.05412695733994979,"3407":0.01451569505429038,"3408":0.00871369463610168,"3409":-0.11576371174887912,"3410":-0.18421669501759605,"3411":-0.22279596458495365,"3412":-0.3371083227843917,"3413":-0.1923447559762503,"3414":-0.012603108882012557,"3415":0.06431871147981752,"3416":0.04289929215360638,"3417":-0.0639938133245524,"3418":0.14021850531446375,"3419":0.28773836070392217,"3420":0.15452003698379596,"3421":-0.14334282795862002,"3422":0.09554198287155548,"3423":0.24092614219881692,"3424":0.27056284258663615,"3425":0.08586396402102679,"3426":0.028437342755912257,"3427":0.0256303223732346,"3428":0.018452605704476262,"3429":0.08164971776446762,"3430":0.05744309702786926,"3431":0.034464877150102255,"3432":-0.08580199579274775,"3433":-0.3993060652174537,"3434":-0.3813580387473863,"3435":-0.052882255720186754,"3436":0.20231756517538246,"3437":0.2721304667249784,"3438":0.398630981914076,"3439":0.13154405180764783,"3440":-0.6783502795623637,"3441":0.02334187834245048,"3442":-0.013175755841917897,"3443":0.015235164239264076,"3444":0.0003605774056782453,"3445":-0.0676396404691055,"3446":-0.0886544847063936,"3447":0.02103157335608374,"3448":0.04703285875772695,"3449":-0.020578799089417787,"3450":-0.002259093385883094,"3451":0.009356799872078514,"3452":0.08098161133201179,"3453":-0.13144700525727115,"3454":0.018357295008360384,"3455":-0.055397823392712604,"3456":0.008521070836879054,"3457":-0.09755818059836902,"3458":-0.05059313139616353,"3459":0.17561427503866842,"3460":0.12035648348234172,"3461":-0.12786943084213315,"3462":-0.046459737189290806,"3463":-0.004770655919892097,"3464":0.03140289437480882,"3465":-0.17632043080435106,"3466":-0.010003003268292966,"3467":0.007349205794236562,"3468":-0.0007847436625574053,"3469":0.010113660200854056,"3470":0.06477461762246715,"3471":0.08068141012402558,"3472":0.011184344770612078,"3473":-0.04081644881653825,"3474":0.011928721860531278,"3475":-0.13588400101563816,"3476":0.08523202143025022,"3477":0.011393604783995821,"3478":-0.053789566970605183,"3479":0.028254431440142176,"3480":0.10689259143896065,"3481":0.06447627215955504,"3482":-0.0029572099292008683,"3483":-0.028844899755833115,"3484":0.19077878637562953,"3485":0.2766524664416652,"3486":-0.06576762400204152,"3487":-0.011084647944362761,"3488":0.09082316512197242,"3489":0.021789968400638605,"3490":-0.15841864521521784,"3491":-0.04740308312514149,"3492":-0.20492917120008758,"3493":-0.049964970384749695,"3494":0.036675448287561044,"3495":-0.267984017918451,"3496":-0.2425136512854579,"3497":0.029229120413149692,"3498":-0.1153364156107007,"3499":0.14974230185137485,"3500":0.03802206543817939,"3501":0.07499831753746532,"3502":-0.3596657308384926,"3503":0.012274373943675561,"3504":-0.0021052822913546153,"3505":0.01427277955116944,"3506":-0.03813890299642944,"3507":0.17434071571188475,"3508":-0.028338959763456198,"3509":-0.018108609481969736,"3510":0.07299760992749886,"3511":0.060008894493430436,"3512":-0.21148363545422919,"3513":-0.02269108682817022,"3514":0.04208920975145491,"3515":-0.033237482239960356,"3516":-0.018434951511005,"3517":0.06823860969287236,"3518":-0.06165052971868508,"3519":-0.01773509943059789,"3520":0.08863329755145433,"3521":-0.09638127232157519,"3522":-0.1927660461111033,"3523":0.23570967292433645,"3524":0.04404594124668577,"3525":0.029706642809543098,"3526":-0.27698011087537694,"3527":0.153703816816355,"3528":-0.03508238387073156,"3529":-0.04179053240380549,"3530":0.09978879949377256,"3531":0.034692100390780675,"3532":0.02642753976426405,"3533":-0.10518759440639626,"3534":0.04124118325076003,"3535":-0.01161159196706311,"3536":0.0646622198596479,"3537":0.027625081861993542,"3538":-0.09538600445679712,"3539":-0.03492025370788816,"3540":-0.04797453806200787,"3541":-0.018905176364992577,"3542":0.0500407462367399,"3543":0.009010201466937092,"3544":0.0536539484358829,"3545":-0.032095267336682304,"3546":0.048587622207207116,"3547":0.050852990187872676,"3548":-0.038312651478157135,"3549":-0.04911658249431152,"3550":0.018936609594430765,"3551":-0.003040375878222609,"3552":-0.016337916237121768,"3553":0.00638407843447989,"3554":-0.020930502711570396,"3555":-0.025511758857267863,"3556":0.01622874996448473,"3557":-0.05264072485875701,"3558":-0.04612914456348657,"3559":-0.0026207607177145304,"3560":0.0021912750659006173,"3561":0.022299968087296975,"3562":0.04711236460942254,"3563":0.027623575661841454,"3564":-0.016685284578962227,"3565":-0.058300127558807126,"3566":0.04064934800908959,"3567":-0.0604552999864526,"3568":0.03185365727266469,"3569":-0.02525518922422568,"3570":0.013657793563310415,"3571":-0.005520365009240244,"3572":0.013988079911750032,"3573":-0.008366252541879584,"3574":-0.025370636809035948,"3575":-0.01618133466758336,"3576":0.04376268284720197,"3577":-0.030107638359376736,"3578":-0.12213888194481165,"3579":0.011236720041524158,"3580":-0.020269566239983553,"3581":-0.0029243049796707867,"3582":0.022859885680310842,"3583":0.02871835328344769,"3584":0.1244897689507183,"3585":0.0148722753833168,"3586":0.03055704279828285,"3587":-0.12680856416387334,"3588":0.1948536016727188,"3589":0.1360746656747263,"3590":0.0241992175172295,"3591":0.03387638961334452,"3592":-0.052324817497271754,"3593":-0.029805272195916496,"3594":-0.06191539840954447,"3595":0.17834331577235493,"3596":-0.05643634728721828,"3597":-0.020104405398854303,"3598":-0.07028128558210754,"3599":-0.025535855471401924,"3600":-0.10722396136958856,"3601":-0.01791228329679695,"3602":-0.02020546114508031,"3603":0.06082879348251344,"3604":0.03366238683906537,"3605":0.0339934849647325,"3606":0.012237128949044286,"3607":-0.002186367868967468,"3608":-0.055053263979543805,"3609":-0.026677502959579533,"3610":-0.015667335823818452,"3611":0.0018383901398960245,"3612":-0.026776722332834072,"3613":0.07190255446597103,"3614":0.02793074055779402,"3615":0.13481731188975396,"3616":0.06853758436786186,"3617":-0.047813742181466626,"3618":-0.17057907289364674,"3619":0.15497638848428721,"3620":-0.07084071303530132,"3621":0.022779755346665646,"3622":0.021267131018359073,"3623":-0.03292536602176809,"3624":-0.008294071642902454,"3625":0.0016295580575939681,"3626":0.08970425796234244,"3627":-0.036230406320202496,"3628":0.0911796185775206,"3629":0.07531033431971991,"3630":-0.08370272763882909,"3631":0.16357380485639356,"3632":-0.0734884828906965,"3633":0.003355924326918,"3634":-0.03716595762798889,"3635":0.1940564518737005,"3636":0.016028041048746323,"3637":-0.05166974040264393,"3638":0.007128563573878966,"3639":0.041411729480368906,"3640":0.1619559702528917,"3641":0.1513327113216483,"3642":-0.14195439983449465,"3643":0.21008825882207982,"3644":0.033909405765246856,"3645":-0.15670505649376623,"3646":-0.08399318437922339,"3647":0.0810983956569687,"3648":0.017029074804328807,"3649":0.18121797024268022,"3650":-0.20975854819035752,"3651":-0.04077506232920978,"3652":-0.1558402981101303,"3653":-0.13604972377414654,"3654":0.15698363230349754,"3655":0.10390726734068058,"3656":0.19216815243123808,"3657":-0.35363664026205316,"3658":0.008163126001169122,"3659":0.048870303134819215,"3660":0.000661971745088252,"3661":0.06491325157644294,"3662":-0.0271179979647406,"3663":0.05985800624354349,"3664":0.044034236195089844,"3665":-0.057154249378963344,"3666":0.01878115786751479,"3667":0.07069776454423748,"3668":-0.01403115859001783,"3669":0.03453828511063671,"3670":-0.03794353884756689,"3671":-0.07294325554562947,"3672":0.040913833618729654,"3673":-0.05342388205415661,"3674":0.048033578719995945,"3675":0.03138419239672978,"3676":-0.00427501135435128,"3677":-0.03888985010543754,"3678":-0.343819530340404,"3679":0.0451666555029827,"3680":0.09309723015427872,"3681":-0.05634168895751511,"3682":0.03431057965817193,"3683":-0.01897856797176322,"3684":-0.0010460674129092445,"3685":-0.04431905830083102,"3686":0.00959917462394339,"3687":-0.06012421775607438,"3688":-0.02653557806228721,"3689":-0.017629336481571042,"3690":-0.02408651465531855,"3691":-0.0441397882730786,"3692":0.0537833566230544,"3693":0.022640354105863125,"3694":-0.03341149906237656,"3695":0.022494970916437502,"3696":-0.012144141931948038,"3697":0.0003637797300802381,"3698":-0.057978888731773225,"3699":-0.00005776955180398061,"3700":0.010415968137987288,"3701":-0.01715350147189879,"3702":-0.08349160368017487,"3703":-0.026270403081541573,"3704":0.030355496200155197,"3705":-0.09065252676387597,"3706":-0.026768799231657217,"3707":0.06543603393017479,"3708":0.06979727736979505,"3709":0.08927826179508036,"3710":0.020492478512835696,"3711":-0.06828269656415736,"3712":0.11667988277043168,"3713":0.04424975586110298,"3714":-0.007852777070171272,"3715":0.02171497414749075,"3716":-0.021180709774179608,"3717":-0.02790942185988935,"3718":0.006995414923814384,"3719":0.09851613796864817,"3720":0.024867270992508923,"3721":0.013545136948300888,"3722":0.06420013710814089,"3723":0.010529754565975318,"3724":-0.018407523007488904,"3725":-0.027971938603588063,"3726":0.04101292154496454,"3727":-0.00779880275688635,"3728":0.024547429440595472,"3729":-0.09536505877069261,"3730":0.011953230949742414,"3731":0.03376314253015671,"3732":-0.04596256239834565,"3733":-0.017769519037925733,"3734":0.05357883085210818,"3735":-0.011865450660956924,"3736":-0.10475432072715063,"3737":0.06880273631647857,"3738":0.0017480258762190407,"3739":0.04118713595236267,"3740":0.28385268975564487,"3741":-0.01924764264953218,"3742":-0.04439241562600149,"3743":-0.08696418646321821,"3744":-0.1269766989376929,"3745":-0.025331846079674068,"3746":-0.037047688006913014,"3747":-0.013467541134559785,"3748":-0.05801222615143111,"3749":0.03638576548842907,"3750":-0.02550549263264588,"3751":0.12891206883294024,"3752":0.02066006368397313,"3753":0.08784667725624176,"3754":0.11141300260225344,"3755":-0.022818046075778146,"3756":-0.125777321056129,"3757":0.0065579533525805455,"3758":-0.0588144337368282,"3759":-0.021165419999924117,"3760":0.01592020554965558,"3761":-0.006026998972610457,"3762":0.0048619331811458714,"3763":0.048152844360004055,"3764":0.16367144829263802,"3765":0.09825584125166546,"3766":-0.07863943637379361,"3767":-0.28196787200308693,"3768":-0.07132615240488319,"3769":0.16566997051923713,"3770":0.2067248652927444,"3771":-0.18290697305061726,"3772":-0.09436003347654322,"3773":0.023552844872924573,"3774":-0.19774514849345154,"3775":0.6643655014050484,"3776":-0.2955893204286069,"3777":0.1818828208371214,"3778":0.3685751365672592,"3779":-0.15781772453864823,"3780":0.12502353202253993,"3781":0.33884225585622924,"3782":-0.020022056985032635,"3783":0.03585255608566346,"3784":0.04543218852441061,"3785":-0.004232209385119546,"3786":-0.05114979861004272,"3787":-0.04256491557508122,"3788":0.050763321792221316,"3789":-0.04433928656694963,"3790":0.014621965169202025,"3791":0.004821414006695586,"3792":-0.032882775218351215,"3793":0.002565001845828624,"3794":0.021789366529439438,"3795":0.016571299982866324,"3796":-0.02353180937537,"3797":0.03140741192813743,"3798":-0.010148681156054011,"3799":-0.0768929166539961,"3800":0.08718155972369154,"3801":-0.08253145040949317,"3802":0.07453250379768167,"3803":-0.024840378800063987,"3804":0.009867162431793976,"3805":0.07261819234312043,"3806":-0.1344013624945467,"3807":0.0001597089038597119,"3808":0.014273339026727245,"3809":0.03426267785470338,"3810":0.006204168404099448,"3811":0.01733877189665845,"3812":-0.020566006073504714,"3813":-0.020695578494677274,"3814":0.05618484655876679,"3815":-0.05950147685409063,"3816":0.061864365636466745,"3817":-0.09760653247729073,"3818":0.03896713218249979,"3819":0.032882229903828594,"3820":-0.08239345656154475,"3821":-0.038805813793167686,"3822":0.1272059593222655,"3823":-0.01683482124426423,"3824":-0.015267199681265492,"3825":-0.036839805364904915,"3826":-0.06792459299269063,"3827":0.05600894900572232,"3828":0.02918825894745812,"3829":-0.012251339527628007,"3830":-0.10708533207088693,"3831":0.12624501337381747,"3832":-0.04713944368520586,"3833":0.06670883582234005,"3834":-0.012537901924094819,"3835":0.08321368321956493,"3836":0.13306270823624822,"3837":0.18636602011471382,"3838":0.03734706302749868,"3839":0.04011993355992208,"3840":-0.07959313917780794,"3841":-0.03925711923767545,"3842":-0.08338841185692969,"3843":0.14381950016609304,"3844":0.001733445697513102,"3845":0.04932986721168159,"3846":0.055998059361777534,"3847":-0.035299657316665824,"3848":0.07846785067428846,"3849":-0.008102775700466837,"3850":-0.007801171681528395,"3851":-0.03207055170887676,"3852":0.011403098570856723,"3853":0.02226957266789165,"3854":0.023329517344832695,"3855":0.013016212652721739,"3856":0.028643181136967105,"3857":0.030084436040242152,"3858":0.034630195213973335,"3859":-0.05502182133992523,"3860":0.045304684403924274,"3861":0.02638420086059417,"3862":-0.10399307674687457,"3863":-0.09015982008962943,"3864":-0.10210051407454535,"3865":-0.039633883983130655,"3866":0.04861632218272075,"3867":0.0425984513988458,"3868":0.15075941288686678,"3869":0.0035465102184973544,"3870":0.005339273499119236,"3871":0.056218544057262766,"3872":0.033705207841780056,"3873":-0.022244607413501816,"3874":-0.04505054693736666,"3875":0.05918675587548579,"3876":-0.07649678535964473,"3877":-0.0715074698588188,"3878":0.009945434779323359,"3879":-0.031986528874753756,"3880":0.05869814785547278,"3881":-0.01306905248560399,"3882":0.05180009090697222,"3883":0.04619967825021755,"3884":0.02531733389969838,"3885":-0.013530970123509487,"3886":-0.04165832431817938,"3887":-0.05871090386388806,"3888":-0.13349692638740465,"3889":0.11984577151652026,"3890":0.06550392106282159,"3891":0.061902701995687426,"3892":0.09771773171200497,"3893":-0.01743007435026473,"3894":-0.0163704546016789,"3895":-0.2736397110491201,"3896":0.01125927188746666,"3897":-0.05402470384817938,"3898":-0.0865598742937158,"3899":0.1981405373651209,"3900":0.01014940870916902,"3901":-0.006280495003408526,"3902":-0.04124539242129119,"3903":0.02187232871259926,"3904":-0.007082313919892329,"3905":0.013952219742697426,"3906":-0.015619284357408052,"3907":-0.011744971088256824,"3908":-0.06617630991741949,"3909":0.1282296873571425,"3910":-0.014299557051394837,"3911":0.06504133007914997,"3912":0.00314446890715519,"3913":0.008530607741105568,"3914":-0.008832865444614597,"3915":-0.03374569431910197,"3916":0.00308596479562536,"3917":0.037246764739262674,"3918":-0.05023027275077414,"3919":-0.08737989901476674,"3920":0.00574318149212671,"3921":-0.12751284444220715,"3922":-0.03633179899223335,"3923":0.11092837023419995,"3924":-0.05418842277827087,"3925":0.009255472251353102,"3926":0.2416277160389501,"3927":0.004292051913940113,"3928":-0.11848804660729483,"3929":0.0711881715192522,"3930":0.16447535937148144,"3931":-0.06132886633680794,"3932":-0.011915433672569645,"3933":-0.07899411293152257,"3934":-0.07570743383409138,"3935":-0.037845515810051994,"3936":0.04422393712122929,"3937":0.02823965762825298,"3938":0.020289902186158983,"3939":0.05384274386318944,"3940":0.009726672129813188,"3941":-0.09363540565889028,"3942":-0.013573500497287637,"3943":0.03623126509594554,"3944":-0.10600333685324706,"3945":-0.08891065098477863,"3946":0.14572407762026532,"3947":0.015465581715951122,"3948":0.05787180640234543,"3949":0.009408705812287695,"3950":0.04351040623847954,"3951":-0.0009112949735727566,"3952":0.008727876988923026,"3953":-0.10977043432724676,"3954":-0.12020529680096972,"3955":0.1270041046036328,"3956":0.22828612589334912,"3957":-0.08066146573578549,"3958":0.03374224911713731,"3959":0.0575451371720167,"3960":0.10804241369765355,"3961":-0.16729248995351037,"3962":0.043449152896416175,"3963":0.007209855307592304,"3964":-0.05125597828888007,"3965":-0.0387961279827048,"3966":-0.020831717393814456,"3967":0.12524266413298732,"3968":0.002366706757902009,"3969":-0.0722257312755077,"3970":-0.034877770031521226,"3971":0.02754192865723106,"3972":-0.15481469939687564,"3973":-0.001980945693912398,"3974":0.0021660055025898624,"3975":0.04310913240975362,"3976":0.07460456865605766,"3977":0.10080844042747526,"3978":-0.019830095135027453,"3979":-0.07350646541664706,"3980":0.0011805601877863242,"3981":-0.03260800020856196,"3982":-0.018054557431432892,"3983":0.06993078361770069,"3984":0.0756630945263574,"3985":-0.0006743848026799884,"3986":0.024212981945580924,"3987":-0.06513399252896737,"3988":-0.21115708566127284,"3989":-0.006386077163763711,"3990":-0.003660050539593872,"3991":-0.003044676575609006,"3992":0.043542594720091574,"3993":-0.03435814140061711,"3994":0.01349345809231295,"3995":0.0012040396650853733,"3996":0.023787427530161016,"3997":-0.006978987368736331,"3998":-0.06533361174198962,"3999":-0.020106268552941436,"4000":0.04674251285603417,"4001":0.030955522821916295,"4002":-0.005411372355645924,"4003":0.09245863939489783,"4004":-0.0592912537113134,"4005":-0.002111543850770531,"4006":-0.022938881932607215,"4007":-0.010504335340301318,"4008":0.028053767284220266,"4009":0.05752589046875932,"4010":-0.003160764095577518,"4011":0.04290674335875085,"4012":0.0011700765313850307,"4013":0.018551968211306263,"4014":-0.04108910305028753,"4015":0.0004493331521574645,"4016":-0.0494266494391241,"4017":0.015832551439470363,"4018":0.07255603349343848,"4019":-0.023475259917541354,"4020":-0.0314762597012552,"4021":0.12093586216754326,"4022":-0.023686631652334374,"4023":0.06418921020700463,"4024":0.01954870926678866,"4025":0.0035132791743796143,"4026":0.03063777525371493,"4027":0.035549521988654874,"4028":-0.036067851121454914,"4029":0.019582068521657718,"4030":-0.10293083240154727,"4031":0.028924494141794387,"4032":0.022171638130319603,"4033":-0.07501582102794883,"4034":0.1118946737142113,"4035":0.011328370546945578,"4036":-0.018044528829892405,"4037":0.07103963784483172,"4038":0.07243405478404108,"4039":0.0368566917426948,"4040":0.07706288082615018,"4041":-0.09342010615741084,"4042":0.06949169159370636,"4043":0.17695886348206835,"4044":0.09488297682271045,"4045":-0.015457771005908009,"4046":0.08739362548761141,"4047":0.09819174706481767,"4048":-0.16931082085334234,"4049":-0.06012686115461788,"4050":0.3153050180790669,"4051":-0.08611760343964785,"4052":0.03720352912600227,"4053":-0.06127641675677413,"4054":-0.13342506990063116,"4055":-0.02224777976950365,"4056":-0.04024531050433843,"4057":0.05870147367990997,"4058":0.04153845330890966,"4059":0.041615134176474076,"4060":-0.20298188238113812,"4061":-0.015536350879756508,"4062":0.024227150024493417,"4063":-0.04178424854729131,"4064":-0.043874985122554465,"4065":0.06763545846098183,"4066":0.051054000497299884,"4067":0.018558505048799898,"4068":0.037908263467473634,"4069":-0.011051318023275178,"4070":0.009132633982915498,"4071":-0.032513392682676875,"4072":0.019594159252847375,"4073":-0.02246828874593785,"4074":0.02021153583123883,"4075":0.03842862382947416,"4076":-0.015538289491488213,"4077":0.029343974431454027,"4078":0.0514056056868876,"4079":-0.0220547155257009,"4080":-0.08096845242220821,"4081":0.27769672442810717,"4082":-0.02035920439580826,"4083":0.01888510808057464,"4084":-0.09331935447414112,"4085":0.02596542260931158,"4086":-0.021067323212237837,"4087":-0.01884732792082458,"4088":0.011180530670815514,"4089":-0.021836774627411265,"4090":-0.0034286257931137785,"4091":-0.08624020016322934,"4092":-0.07270732431135613,"4093":0.0599119522929983,"4094":0.038361663263588204,"4095":0.05307642372119036,"4096":-0.10155947123672882,"4097":-0.004198326529899753,"4098":0.013665081740774437,"4099":-0.010221164423545506,"4100":0.015458555723340145,"4101":0.021469936456711892,"4102":0.04483741761744247,"4103":-0.05129522239136525,"4104":0.04807724132825949,"4105":0.042427333293316945,"4106":0.008038648733560548,"4107":0.03620518992312923,"4108":0.048384843732584185,"4109":0.03981683684704862,"4110":-0.04402800015142918,"4111":0.07688477682455348,"4112":0.13361320463160678,"4113":-0.06208822646263603,"4114":-0.025294816624930362,"4115":0.20463008526293408,"4116":-0.24366919142936527,"4117":-0.024876221497090976,"4118":-0.02209305968047171,"4119":-0.0033691904276687184,"4120":-0.029694844059099473,"4121":0.006262721794080112,"4122":-0.04438522684473203,"4123":0.047880714512191926,"4124":0.02672577655784885,"4125":0.007484585659785904,"4126":0.1469496327828043,"4127":-0.07928760362305777,"4128":0.02413916398236372,"4129":0.05601102404159135,"4130":-0.07143111013216262,"4131":-0.11019967997025136,"4132":-0.020480001403050303,"4133":-0.031940138459574295,"4134":0.06630430333243537,"4135":-0.10207695288865541,"4136":-0.10129353975436406,"4137":-0.07281269395286383,"4138":0.029404656303021616,"4139":-0.13421796859794663,"4140":-0.011098001147366449,"4141":0.12345796299088167,"4142":0.0018147499214071902,"4143":0.12230328589847049,"4144":-0.08512840450452538,"4145":-0.046400140838315534,"4146":0.15683799318750918,"4147":0.1712936598861694,"4148":-0.018869239626212106,"4149":0.04001060393579657,"4150":-0.11124905904054272,"4151":-0.1500829764451134,"4152":-0.05174398854972014,"4153":0.2062616896422121,"4154":-0.09129912799972482,"4155":-0.13173902522420586,"4156":-0.19204265476503585,"4157":0.45526455862989257,"4158":-0.8417248335973777,"4159":-0.08737483658978888,"4160":-0.0795980290843402,"4161":0.1866325178176406,"4162":-0.0793801298325693,"4163":-0.5320388414968169,"4164":-0.19579164365388482,"4165":-0.11609688590993136,"4166":-0.5786701686459609,"4167":-0.7771762363365355,"4168":-4.9201868870245615,"4169":-0.23040643146192685,"4170":-0.36893547535777843,"4171":0.011436640608300203,"4172":0.23133804707600275,"4173":-0.38899278652264047,"4174":-0.15653785501398265,"4175":-0.40874756766374665,"4176":-0.2860202055237458,"4177":0.7188533309325364,"4178":1.109849207417973,"4179":0.15490042909465376,"4180":0.3265075862950064,"4181":-0.2124420173872518,"4182":-0.24051486172817008,"4183":1.355694809975445,"4184":0.2611763253694798,"4185":0.028036183973430464,"4186":-0.03836409087464052,"4187":-0.023619325875165624,"4188":-0.0703000583620093,"4189":0.030352583655128294,"4190":0.031082936922141216,"4191":-0.035476338813828674,"4192":0.02219003236759254,"4193":0.017515542823677392,"4194":-0.008662298135737755,"4195":-0.018408174236940904,"4196":0.018040937826110073,"4197":-0.018713447045591004,"4198":-0.016670998747291395,"4199":0.004348183092459558,"4200":-0.00273126541261894,"4201":-0.01749789952591387,"4202":0.039226621403224096,"4203":-0.02592077222142974,"4204":0.02628383689396204,"4205":-0.14376750748907707,"4206":0.02023843495038883,"4207":-0.04452006287193958,"4208":-0.0907211912915466,"4209":-0.05408583365802556,"4210":0.0012858365134818094,"4211":-0.0016586429172097536,"4212":-0.00768783160857416,"4213":-0.016188654192373843,"4214":0.019893964942349563,"4215":-0.01790862295915277,"4216":-0.0184987964602662,"4217":-0.07948954169522991,"4218":-0.029746709632854397,"4219":-0.03286462659414373,"4220":-0.06854654906223195,"4221":-0.003266863629970686,"4222":0.002750859328328946,"4223":0.06445616876403416,"4224":0.05348683156801207,"4225":0.079776537801982,"4226":0.08798461148675765,"4227":-0.10476721066057812,"4228":0.09810978115287579,"4229":0.0697452813136456,"4230":-0.012103315256252202,"4231":-0.005798878696661768,"4232":0.05898126675055927,"4233":0.04531762137550057,"4234":-0.07508364061374692,"4235":-0.059732826152346305,"4236":-0.08658487618684726,"4237":0.0988529950369163,"4238":0.040502713320412984,"4239":-0.18600571013732156,"4240":-0.06989856240176523,"4241":-0.0032895086762322065,"4242":-0.004508754102174269,"4243":0.06052584358732662,"4244":0.04048230268032246,"4245":0.009705989942690722,"4246":-0.14928150499866158,"4247":0.03550465114982802,"4248":-0.03815723592115583,"4249":0.03425890401861284,"4250":0.0033399312409170003,"4251":0.013250789360047938,"4252":-0.007196373461417615,"4253":0.023352742953217872,"4254":0.048507249802749376,"4255":0.04349991833053919,"4256":0.047961481069831384,"4257":0.04414477777936695,"4258":-0.04070645430610604,"4259":-0.0008263905252477478,"4260":0.11004061858828056,"4261":-0.08543547746091025,"4262":0.08467328783616987,"4263":0.011673864719063941,"4264":0.009587160772067988,"4265":-0.04149301028542602,"4266":-0.01977862561294686,"4267":-0.28272156331865517,"4268":-0.002884212218528872,"4269":0.048107941838703164,"4270":-0.05502617826643498,"4271":-0.024180503448531756,"4272":-0.006642333897988865,"4273":-0.02473457059127964,"4274":0.014441733545984235,"4275":0.059834887911733824,"4276":0.05042937552948761,"4277":-0.10810848828079563,"4278":0.0029181323743876545,"4279":0.010388090626204997,"4280":0.014041461843665507,"4281":-0.01799563079545155,"4282":-0.0036962024269648203,"4283":0.039853859533213555,"4284":0.018737194142494387,"4285":0.0029695174623230593,"4286":0.008975292930512501,"4287":0.019443211231599346,"4288":-0.04921040062483431,"4289":-0.012483671676134468,"4290":-0.002514662451139478,"4291":0.04680301264152883,"4292":0.01934413603851872,"4293":0.010496576679209228,"4294":0.059328051665351406,"4295":0.0002652302465328608,"4296":-0.004224168876110579,"4297":-0.14892435383020525,"4298":-0.20324987158975488,"4299":0.04461325230541115,"4300":0.08855630470133156,"4301":-0.19927146203960475,"4302":-0.16951423433138174,"4303":-0.03709068531866326,"4304":-0.05486005381880206,"4305":0.0190204713962281,"4306":0.0014582695856951127,"4307":0.03277822899293743,"4308":-0.21267801137333228,"4309":0.17976249145691847,"4310":0.35044931967684334,"4311":0.07502265636189874,"4312":-0.5719781188896947,"4313":0.6203518796662236,"4314":0.0642758464300472,"4315":0.21527348794887038,"4316":-0.17674556648916961,"4317":0.039859765038669476,"4318":0.4100135618104423,"4319":0.28475652455755185,"4320":-0.17888695767772925,"4321":0.7796984661671507,"4322":1.2813009289909918,"4323":0.9827672250534178,"4324":-0.19697248489644598,"4325":-0.31455738636302777,"4326":0.01990126648228911,"4327":-1.1160527960749167,"4328":-0.6815422353596294,"4329":0.007099431604495133,"4330":-0.17639294038322603,"4331":1.0976296722872276,"4332":0.32648281471548923,"4333":-0.0799342626301556,"4334":-0.5214608907496264,"4335":-0.34998889901724467,"4336":-0.39370970559204677,"4337":-0.2556833423794615,"4338":0.40566883009808213,"4339":0.20537756598833948,"4340":-0.006675666951097327,"4341":0.01814699270044322,"4342":0.003937089739051405,"4343":0.0026870992748661616,"4344":0.09873100221698511,"4345":-0.007534364459388361,"4346":0.07005672824891898,"4347":-0.020130708252318722,"4348":-0.013162916911655704,"4349":0.03482314075194458,"4350":-0.0014515073238769091,"4351":0.008347488541674594,"4352":-0.006878628602843671,"4353":-0.033457778343500964,"4354":0.02838523754833729,"4355":0.09016068760552205,"4356":0.007627340570397309,"4357":0.014007190581276847,"4358":-0.0014097037990467177,"4359":0.009177819263150217,"4360":0.03236139234209748,"4361":-0.007471734118202956,"4362":-0.023478300365755177,"4363":0.09624835713658259,"4364":-0.09066959454782444,"4365":-0.0010722383559498368,"4366":-0.0037414773436197923,"4367":0.002255284230005857,"4368":-0.03310836995027696,"4369":0.014386633385633606,"4370":-0.0542036081517553,"4371":0.01993765275844655,"4372":-0.017421264791928234,"4373":0.09009981017893287,"4374":0.02108717534488932,"4375":-0.054340423766165526,"4376":-0.07415539250238867,"4377":-0.005253806223034201,"4378":0.016578042277151112,"4379":0.03985535606523851,"4380":-0.02841148668787391,"4381":-0.027889805941826985,"4382":0.010203794396761553,"4383":-0.006458075250510979,"4384":0.10655070494080231,"4385":-0.0447649964743545,"4386":0.006089867782054347,"4387":0.06502239756964268,"4388":-0.02377275861251174,"4389":0.04188813496375956,"4390":-0.011834885440870298,"4391":0.25524686355891446,"4392":-0.1021155836282316,"4393":0.03492821092913876,"4394":-0.03241373523558822,"4395":-0.16467658904297233,"4396":-0.07975395042701373,"4397":-0.029097336521767634,"4398":0.04638906529745657,"4399":0.050133156195225514,"4400":0.04620450702766385,"4401":-0.06279359688968882,"4402":0.02737201968761042,"4403":0.10154068530058018,"4404":0.1277060443962751,"4405":0.05699865244551258,"4406":0.049226190193355945,"4407":0.03653738507498053,"4408":0.05756685362577035,"4409":-0.06353798943249708,"4410":-0.020983926887569303,"4411":0.032018503815288805,"4412":-0.0014209037685283044,"4413":0.011006387288449916,"4414":-0.046443288041158576,"4415":0.02062355921086312,"4416":-0.026509794262076794,"4417":0.06968626613318356,"4418":-0.05294283005571264,"4419":-0.008624196410579293,"4420":0.006586863285526464,"4421":-0.018262383293093237,"4422":0.2508078273634454,"4423":-0.12528408289271337,"4424":0.01688131196882375,"4425":0.1732183300563599,"4426":0.27073075249585177,"4427":-0.041631902196153654,"4428":0.007147626900279588,"4429":-0.030860951862369555,"4430":-0.12037198320530898,"4431":0.002012323136315198,"4432":-0.009521809521886202,"4433":0.028672967409817993,"4434":-0.008715144290059637,"4435":0.0074789098782416085,"4436":0.08176040073910011,"4437":-0.1273112121808178,"4438":0.05890980293993955,"4439":0.059773243266330385,"4440":-0.007245331137416033,"4441":0.024462950234519755,"4442":0.09118526584621112,"4443":0.02181876092579951,"4444":-0.03263688426303197,"4445":0.03174432205600529,"4446":0.05293854565220884,"4447":-0.12197627567422645,"4448":-0.0031812787277567794,"4449":-0.026706840351047558,"4450":-0.05040235947884877,"4451":0.002161876207786381,"4452":0.10695242599710017,"4453":-0.13194564077681897,"4454":0.17761645532493983,"4455":0.0567327922903177,"4456":-0.11713007528448688,"4457":0.0652607124570792,"4458":-0.007104584318449618,"4459":-0.030331695583454537,"4460":-0.05046351804903127,"4461":-0.00514988693764974,"4462":-0.01649673114343121,"4463":0.003031405548353135,"4464":0.03135180874772799,"4465":0.035282517732229464,"4466":0.03330455465833064,"4467":0.01038443698036521,"4468":0.010488559796128307,"4469":-0.019515353380944362,"4470":0.01230674525841057,"4471":-0.04289808070120899,"4472":-0.027905996842798795,"4473":0.003392185074357808,"4474":0.02922179468975166,"4475":0.031727283797441864,"4476":-0.05456217634802935,"4477":-0.03736813351324755,"4478":0.027483836982347114,"4479":0.0320535187048104,"4480":-0.07075000877599735,"4481":-0.06539918680088914,"4482":0.05016859438186902,"4483":-0.10697851954013589,"4484":-0.41213048018010295,"4485":0.05367037037914922,"4486":0.049899175789000244,"4487":-0.045297926506118254,"4488":0.023538996057546776,"4489":0.009624289419475558,"4490":0.01211675768741356,"4491":-0.004949756753343228,"4492":0.017652642246995493,"4493":-0.028513182896228853,"4494":0.02542367667829168,"4495":-0.0312957551171064,"4496":0.027959752883453776,"4497":0.04654779003147502,"4498":0.002551234496427566,"4499":-0.040829906586648215,"4500":0.022126478682668167,"4501":0.02491171652419458,"4502":-0.03523604272406545,"4503":0.041194611370380066,"4504":0.06299565170324543,"4505":0.004486189112796357,"4506":0.057737882654679586,"4507":-0.04285369228258544,"4508":0.054755693241775906,"4509":-0.0013495374718953885,"4510":-0.05111394942497685,"4511":0.04529663892139886,"4512":0.13372957909753375,"4513":-0.1456253347703459,"4514":-0.11508761405013901,"4515":0.0009258708493241699,"4516":-0.01791731787692801,"4517":-0.052094035851980705,"4518":0.10247049438606999,"4519":0.10877246914296235,"4520":-0.01664639144387073,"4521":-0.008799566527422826,"4522":0.0009971646915682161,"4523":-0.002798750650654475,"4524":-0.001752351465179456,"4525":-0.06909319730671946,"4526":0.0034873801912984987,"4527":0.07129970892113692,"4528":0.05580138826875517,"4529":0.05227147727839546,"4530":-0.0330913236052779,"4531":-0.02492698815609245,"4532":0.03212085712007967,"4533":-0.05944196188651291,"4534":-0.0015740534237700574,"4535":-0.08755180744537847,"4536":0.007242607134253022,"4537":-0.008341069561754223,"4538":0.029508778718138902,"4539":-0.05131883427229081,"4540":0.06618035324779342,"4541":0.0257955698382766,"4542":0.03122501397152504,"4543":-0.005950198450495352,"4544":-0.04456170208275409,"4545":-0.20326765794602156,"4546":-0.28792487425554153,"4547":0.024150879009989305,"4548":0.0342229495984384,"4549":-0.09369004560163639,"4550":0.06505789894159114,"4551":-0.046600590235395756,"4552":-0.013590095576523686,"4553":0.031390054957818526,"4554":0.012101607433520001,"4555":0.013881277584221094,"4556":-0.08450518177524649,"4557":0.1559999929586112,"4558":2.1513792263973603,"4559":2.741926546509578,"4560":0.3783810336207877,"4561":0.4042913000795142,"4562":-0.32176163165426247,"4563":1.8374567318898312,"4564":-2.0005478113044566,"4565":-0.886875079014801,"4566":-3.5879607418463566,"4567":-0.8418836078954662,"4568":1.638968455754324,"4569":-0.9740498486207768,"4570":-3.9535669460139546,"4571":-3.7999343750605945,"4572":1.1858529330037415,"4573":-0.4663366210826426,"4574":-3.784320737299783,"4575":-5.173579062085689,"4576":-3.9637610530022385,"4577":1.3142115641703545,"4578":-4.547868772638552,"4579":-2.4808108349104594,"4580":2.3276094395889224,"4581":0.991404261203136,"4582":2.0542117549459293,"4583":1.747734718255283,"4584":-7.1283367137603335,"4585":6.171800081442508,"4586":-9.289612446124226,"4587":4.667417123587665,"4588":0.048863023065665145,"4589":-0.05046786345626961,"4590":-0.010261459400911504,"4591":0.0043842425744381195,"4592":-0.05048450560402645,"4593":0.03573229675290467,"4594":-0.021321604963932934,"4595":-0.02826682894382772,"4596":-0.007980224904247137,"4597":0.21429091932924413,"4598":0.029412204928243948,"4599":-0.09919949990615384,"4600":0.05167894623561191,"4601":0.03267401123566773,"4602":0.0053373672516365605,"4603":0.03281882482551335,"4604":0.11164234289073499,"4605":0.003261818165509199,"4606":-0.07582835317990118,"4607":-0.047368004629419425,"4608":-0.18690960622836594,"4609":-0.0879926043813882,"4610":0.07040139646572234,"4611":0.15955250847142216,"4612":0.20524073598591427,"4613":-0.003975969019391437,"4614":0.00079402412870246,"4615":-0.02245175954331218,"4616":0.010367165570767837,"4617":-0.038384016353146554,"4618":-0.040575725665767975,"4619":0.06717446815321389,"4620":0.04462176526187885,"4621":0.04452715768083606,"4622":0.020034579183889276,"4623":0.05168059224438785,"4624":-0.0022859760485205408,"4625":0.0628235193275857,"4626":-0.08311286851466115,"4627":-0.08332345461910015,"4628":-0.08421235534510126,"4629":-0.054631108041666286,"4630":0.10785367585890718,"4631":-0.044771284179384994,"4632":-0.08978927244906404,"4633":0.023533806649842514,"4634":-0.013196751533021986,"4635":-0.053274503723717184,"4636":-0.03597016254967579,"4637":0.05201235170879018,"4638":-0.054978317693600856,"4639":-0.11290193562097295,"4640":0.04753946867204342,"4641":0.010365435672550913,"4642":-0.10114975771073373,"4643":0.19997572618535128,"4644":-0.020555970061108878,"4645":-0.0004293793513997868,"4646":-0.005761526624319396,"4647":-0.009873819411145918,"4648":-0.00765110639758415,"4649":0.09584748395289377,"4650":0.048816995561180854,"4651":-0.02069246200605189,"4652":0.07863392229745694,"4653":0.005145775255805254,"4654":-0.009188935801055462,"4655":-0.03852329068051751,"4656":-0.06299579733391274,"4657":-0.014482715189066608,"4658":-0.02746107886890913,"4659":0.0023493569112278518,"4660":0.052219470501562115,"4661":-0.01765345022626339,"4662":0.022942561500007102,"4663":0.14604504491966278,"4664":-0.019656367888440295,"4665":-0.0705081419016656,"4666":0.009782589175379653,"4667":-0.04850209848765174,"4668":-0.02814811314219908,"4669":0.06607629361324022,"4670":0.13561048557416705,"4671":-0.08472491338568999,"4672":0.22164679544322,"4673":-0.24120222703860236,"4674":0.1588962197281446,"4675":-0.017026691623307225,"4676":-0.013336283251542638,"4677":-0.0023406111219166172,"4678":0.05919116435406859,"4679":-0.004891603555056921,"4680":0.02648459285969929,"4681":0.05386197375327464,"4682":0.020968436904065874,"4683":-0.010533613581347115,"4684":0.1047808640313001,"4685":-0.03863406758680584,"4686":-0.006195012634441693,"4687":0.03166183795458479,"4688":-0.07947734574377861,"4689":-0.03374004192406964,"4690":-0.11149317662403244,"4691":-0.02728399182277587,"4692":0.07744339967373125,"4693":-0.06224719412366249,"4694":-0.15589986332507846,"4695":-0.0758020274602431,"4696":-0.018266177989849385,"4697":-0.15618311021102274,"4698":-0.023981820724401727,"4699":0.097266391175752,"4700":0.053805309826245755,"4701":-0.08575598593746529,"4702":0.0637038605941171,"4703":-0.05110452916575395,"4704":0.0028418174539978015,"4705":0.14805413819274704,"4706":0.026080297842860537,"4707":0.038392159846357285,"4708":-0.041344335855184405,"4709":-0.04710614790534771,"4710":-0.020584006455377654,"4711":0.19728269333093737,"4712":-0.038653277656229494,"4713":-0.05015487313291817,"4714":-0.03235188496285451,"4715":-0.05953818867226087,"4716":-0.0092915186275546,"4717":-0.047711339137174756,"4718":-0.0007125053134216285,"4719":0.060406486648814976,"4720":-0.050322858619495474,"4721":0.13377329686923875,"4722":0.03832814067920088,"4723":-0.03519785760267228,"4724":0.0667831814970096,"4725":0.08152772784979595,"4726":-0.06642047921221797,"4727":-0.04565331269966191,"4728":0.07199693906885178,"4729":-0.0457917773690921,"4730":-0.02628482051596221,"4731":0.0029737731426673244,"4732":0.0930735546307313,"4733":0.0023184334132668087,"4734":0.04380673038111781,"4735":0.05471691241477313,"4736":0.0745991239101245,"4737":0.03151464295421902,"4738":0.018005161784148814,"4739":0.03416891268790587,"4740":0.014629002631298418,"4741":-0.05460772108147918,"4742":0.0023849147822965536,"4743":0.032359393212159054,"4744":-0.07375759815189446,"4745":-0.02047551438446286,"4746":-0.039240776887595655,"4747":-0.026885014600415355,"4748":-0.03199606636684025,"4749":-0.042914789099825525,"4750":0.07065819888909829,"4751":-0.029142027419130336,"4752":-0.04695723766786675,"4753":-0.005652184530788332,"4754":-0.00454304481276591,"4755":-0.019688013991275816,"4756":-0.022243379307340577,"4757":-0.04624012074895653,"4758":0.042556118260359245,"4759":-0.008236918393881592,"4760":0.016662593572783765,"4761":0.026865426174605668,"4762":0.035688013520166914,"4763":0.025594986526897536,"4764":-0.022307120642854374,"4765":-0.09500275224963908,"4766":0.05356134904979112,"4767":0.19396288815699522,"4768":-0.013243233885434913,"4769":-0.00619679509575823,"4770":0.014012839514232267,"4771":0.0245670852239315,"4772":0.012047524122566489,"4773":0.08309917846097993,"4774":0.048937415372581054,"4775":-0.03908665543083651,"4776":-0.028766402345651823,"4777":-0.12342685761149667,"4778":0.1063092538030667,"4779":0.02189742675684569,"4780":-0.013124506871282247,"4781":0.0703844077324404,"4782":-0.0007681572643102502,"4783":0.13796762119821307,"4784":0.023723271559283575,"4785":-0.03424398632659588,"4786":0.029751199263514865,"4787":0.08307947837817523,"4788":0.05418175574052345,"4789":0.012791019272280038,"4790":0.06344200371904495,"4791":0.025093327547824985,"4792":-0.09605216590590362,"4793":-0.04539895877775287,"4794":-0.21111974160379127,"4795":-0.0033355305350804216,"4796":0.12374603349258397,"4797":-0.1889574764741485,"4798":-0.02420304775269183,"4799":0.037786773118842515,"4800":-0.02229548526332419,"4801":0.03433840065869595,"4802":0.04395321188331236,"4803":0.024230610164269105,"4804":-0.13996282848964847,"4805":-0.010603445365826544,"4806":0.0005153134177702263,"4807":0.00365635153387787,"4808":0.023355787673163902,"4809":0.06759582716057116,"4810":-0.02832788990750659,"4811":-0.0010123595734872154,"4812":-0.00092005040101841,"4813":0.010111893250252645,"4814":-0.09199016797142992,"4815":-0.014932136263734295,"4816":0.04234009850376031,"4817":-0.0022090735499152104,"4818":-0.03436443904037023,"4819":0.012564811834156607,"4820":-0.017318368427401927,"4821":0.0041836855880098755,"4822":0.03154945695171878,"4823":-0.05428524988224255,"4824":0.04437191156493745,"4825":-0.06385541006707315,"4826":-0.001472009619614821,"4827":-0.06685372839581444,"4828":0.09791148165566074,"4829":0.17437238178372494,"4830":0.00915070259149463,"4831":0.014084635351114925,"4832":0.005490478944656292,"4833":0.011368989584037847,"4834":-0.00729410280662196,"4835":0.09422519704471635,"4836":-0.02243349263695654,"4837":-0.046193078944187206,"4838":-0.02528331806465654,"4839":-0.0059002752147097395,"4840":-0.06483469867956099,"4841":0.00611049512568833,"4842":-0.04401805624590897,"4843":0.035373632069391266,"4844":-0.008357607340339258,"4845":-0.023168534176863816,"4846":0.014885596849260082,"4847":-0.030854201078236408,"4848":0.03057743804111761,"4849":-0.01068749710064194,"4850":-0.007183032826522356,"4851":-0.06204197038359063,"4852":-0.022267322448810946,"4853":-0.0343514615231033,"4854":-0.00008622991369380806,"4855":0.028309393055393976,"4856":-0.016897345203219845,"4857":0.04383235369856714,"4858":0.01507372565328634,"4859":-0.10412016706473938,"4860":-0.07450011988980794,"4861":-0.017594774407461104,"4862":0.0017710108672629673,"4863":-0.001660954133076464,"4864":-0.014605050572636802,"4865":-0.004242396546778456,"4866":-0.003476940383828198,"4867":0.04345475459324242,"4868":0.12070626397864057,"4869":0.04336680829775674,"4870":0.0980160776462177,"4871":0.006091082167950218,"4872":0.002477891097912602,"4873":0.013096110959253471,"4874":-0.10045519730765136,"4875":0.03669526972341235,"4876":-0.16038920012681931,"4877":-0.09159116829454166,"4878":0.13532289183786753,"4879":-0.06429347544494779,"4880":-0.1125245968625364,"4881":-0.03579326283914461,"4882":-0.007300495249114554,"4883":-0.09703547138993861,"4884":-0.11053440599639223,"4885":0.02104086600765703,"4886":0.05588601310960698,"4887":-0.06864199495336895,"4888":0.020995473819249714,"4889":0.04391016721497749,"4890":-0.026656880126761474,"4891":0.25035902731069004,"4892":0.00043711255170343757,"4893":0.037844474261810694,"4894":-0.022915877093458912,"4895":-0.008907419794292042,"4896":-0.028701976781179184,"4897":0.19098776412996937,"4898":-0.047591875492429485,"4899":0.06460865880670155,"4900":0.034604215010994845,"4901":0.013898896727529116,"4902":0.11801451006594119,"4903":-0.008526199564497874,"4904":0.0019471068588690302,"4905":-0.05503260327327606,"4906":0.004798364439446358,"4907":0.013055521190659204,"4908":0.0007099847597894168,"4909":-0.006185117217353004,"4910":0.06253065953170672,"4911":0.017897267972398358,"4912":0.009195940749641637,"4913":0.0365717375560812,"4914":0.06999823993264406,"4915":-0.045665550757658255,"4916":0.016619733072262263,"4917":0.0666130755661468,"4918":0.11833354203435968,"4919":-0.07105103662445048,"4920":0.06202747533784882,"4921":0.11687948731695393,"4922":-0.11617301444086428,"4923":0.005018884291427504,"4924":0.004387798519531134,"4925":0.017785978162319985,"4926":0.013238525485046896,"4927":0.0009710438634724292,"4928":-0.03384737870543779,"4929":0.0140976782987256,"4930":-0.04869483953978832,"4931":-0.032207193358510466,"4932":0.021104304870823914,"4933":-0.08123916432801355,"4934":0.024575407341500603,"4935":-0.015341402643457396,"4936":0.03093097781315407,"4937":0.030621992539354295,"4938":0.11956000034100701,"4939":0.01710489252043169,"4940":0.005357500724096752,"4941":-0.022162068407212483,"4942":0.07237802280393918,"4943":-0.09207253252375003,"4944":-0.044204460740216056,"4945":0.05154619179545233,"4946":-0.002855927924530834,"4947":0.04925559608041794,"4948":0.032092128100562556,"4949":0.20489125432047867,"4950":0.026065671711113732,"4951":0.01810824771203303,"4952":-0.037695612769661574,"4953":-0.04241193858927636,"4954":-0.00941019672368555,"4955":-0.012682969679161575,"4956":-0.035661483007493354,"4957":0.0017950533522775093,"4958":-0.0022591829024253512,"4959":-0.02758137132827203,"4960":0.08627936954207606,"4961":0.004815131218985829,"4962":0.04859656262919326,"4963":0.08161662024251648,"4964":-0.021554031219145713,"4965":0.006032774957037213,"4966":0.01339602710460753,"4967":-0.04248079361414698,"4968":0.03358469253414992,"4969":-0.1236012289486409,"4970":-0.02509544672999725,"4971":0.02355303302729568,"4972":-0.0017758186769582659,"4973":-0.1135631356472399,"4974":0.03338689735345206,"4975":-0.02542576364544915,"4976":0.04432985921815378,"4977":-0.05002129404373865,"4978":0.011077529511653063,"4979":-0.09187965996479877,"4980":-0.02321546311086557,"4981":0.008021851121996526,"4982":0.09139195006779238,"4983":-0.13131535799386812,"4984":0.26178687466203576,"4985":-0.06545330035964238,"4986":0.0067904149127666034,"4987":0.009931364705524262,"4988":0.018941149622146086,"4989":-0.007087142005297935,"4990":0.022255729198699722,"4991":-0.042819826961393734,"4992":0.06566625846142006,"4993":0.06570153121909629,"4994":-0.028910432378362796,"4995":0.08412058107299371,"4996":-0.06804718491330483,"4997":-0.01032479841108358,"4998":-0.03632717133315398,"4999":-0.059692903058787015,"5000":-0.11690687564469723,"5001":-0.004356456654115919,"5002":0.04838559595515054,"5003":-0.04168918602575495,"5004":-0.039827110047596724,"5005":0.0769173692134588,"5006":-0.010074602035919937,"5007":-0.062357584636163914,"5008":0.011597478925620294,"5009":0.05988958468332302,"5010":0.022513603286958072,"5011":-0.004589926584133971,"5012":-0.07651687029945385,"5013":-0.06490310612036887,"5014":0.10275345677781403,"5015":-0.1709517669072807,"5016":0.016396511441240817,"5017":0.015246710649563637,"5018":0.010874004879835914,"5019":-0.0015005943352970495,"5020":0.027640881769049896,"5021":0.054889358903562606,"5022":0.024683012487404343,"5023":-0.03678506075069048,"5024":0.0081972549604672,"5025":-0.05191514953233481,"5026":-0.044628769975984466,"5027":-0.011259819147879097,"5028":-0.06566647969380096,"5029":0.0523620233134439,"5030":0.03904790342072534,"5031":0.1167765272494253,"5032":0.004580993995860233,"5033":0.03171378222978298,"5034":-0.053293373060605734,"5035":0.13794853372392404,"5036":-0.07844917357324376,"5037":-0.020163156178091755,"5038":0.006288740205195584,"5039":0.04278374905434995,"5040":-0.014610129429881115,"5041":0.10237343109751737,"5042":-0.1532038405206043,"5043":-0.09426452525685357,"5044":0.002596893578059883,"5045":-0.08172919082163768,"5046":-0.17496071284921702,"5047":-0.008853995389278013,"5048":-0.030899496429324748,"5049":-0.015646718978362548,"5050":0.04590628678436189,"5051":0.06046876629316018,"5052":-0.0642713123098644,"5053":-0.015503338789332731,"5054":-0.036472252648237335,"5055":-0.05936235645058928,"5056":0.03651643951416535,"5057":-0.21180134896524191,"5058":-0.010730665185624964,"5059":-0.0420293553640021,"5060":0.009617258772513007,"5061":-0.008074660074717629,"5062":0.02869987567543615,"5063":0.0047023611227776764,"5064":-0.02065432395405972,"5065":0.005410423198445088,"5066":-0.02160245245628289,"5067":-0.03138490833251449,"5068":-0.01502487653499449,"5069":0.00040822075158628056,"5070":-0.052726344028245456,"5071":0.02405495081096986,"5072":0.04196709354573759,"5073":0.17344919783039783,"5074":-0.012715585478772487,"5075":-0.037506075777536924,"5076":0.16578783607519687,"5077":0.26044547354629055,"5078":0.008545237936305173,"5079":0.07680822543944142,"5080":-0.04848775969184238,"5081":-0.030505694569942468,"5082":-0.06352085994965422,"5083":0.21643056646986175,"5084":-0.04485749090322807,"5085":-0.026214026516585713,"5086":0.0038595419159262326,"5087":-0.1111975042642887,"5088":-0.0010909473434088702,"5089":-0.011974107478786243,"5090":-0.06063549037311256,"5091":0.03640722962765843,"5092":0.04922801535815864,"5093":0.14796765366870768,"5094":0.022746671623949992,"5095":-0.11935654090021207,"5096":0.08936058503596782,"5097":0.19614698505516445,"5098":-0.05455856704246775,"5099":0.10841908030293727,"5100":0.07404348844255569,"5101":0.0013186615056170977,"5102":-0.09720323699186897,"5103":0.14262431538226267,"5104":0.02535079889362715,"5105":-0.13018814158008302,"5106":0.004951112560794836,"5107":0.11560864222989828,"5108":-0.17996424292284965,"5109":0.02718375495366529,"5110":-0.033912036294079996,"5111":0.006895893164447083,"5112":0.011162628263669622,"5113":0.040771001098759356,"5114":-0.10190269935407366,"5115":0.013984502875233444,"5116":-0.020990835387616223,"5117":-0.009484066923137495,"5118":0.04646791386875222,"5119":-0.018673139299475103,"5120":-0.002253211356532204,"5121":-0.08043051823055741,"5122":0.02964214236197718,"5123":-0.008139147251873913,"5124":-0.1200264048316647,"5125":-0.04414236165448067,"5126":-0.01232288188123493,"5127":0.008531230808557366,"5128":-0.018927177364686652,"5129":-0.015155056001951675,"5130":0.0315809449625091,"5131":0.04390953797184233,"5132":-0.0673182109579075,"5133":0.05882610814545771,"5134":0.09215655079124839,"5135":-0.06755842247101297,"5136":-0.00906790182156622,"5137":0.006014567854479496,"5138":-0.023232500012587285,"5139":0.09563312701351175,"5140":0.0024646332538128383,"5141":0.0007720756693799516,"5142":-0.015407178552512786,"5143":0.05030581148709208,"5144":-0.009077300599645802,"5145":0.1322104256738009,"5146":-0.028495942486735155,"5147":-0.005441594802224847,"5148":-0.05945172638088168,"5149":-0.06116818785053877,"5150":0.025265800457967807,"5151":0.01572053255593243,"5152":-0.024531989816867866,"5153":0.06246604071581342,"5154":0.01578133857015705,"5155":0.1603834696294954,"5156":0.052126276031383326,"5157":-0.028085809275134502,"5158":0.0272618465260584,"5159":0.0962831390508449,"5160":-0.053712751985741686,"5161":-0.009515780864584869,"5162":0.07516108567951885,"5163":0.023600467194834003,"5164":-0.057349376894629656,"5165":0.06250120872645391,"5166":-0.03327008081584459,"5167":-0.015298453037601387,"5168":0.014537051382959205,"5169":0.00888165088960254,"5170":0.03567590724812408,"5171":0.04428105662198824,"5172":0.005026763965280744,"5173":0.001851118887131654,"5174":0.014449813034192481,"5175":-0.02505759195204868,"5176":-0.03273810324787817,"5177":0.0325833969098276,"5178":-0.001500269534084043,"5179":0.06116890816278771,"5180":0.07719489523681004,"5181":0.07459508034414741,"5182":-0.03932496284651327,"5183":-0.00637888656089759,"5184":-0.025054773024489326,"5185":0.012163188328826118,"5186":-0.10397551919642607,"5187":-0.006861174691515538,"5188":0.031209157773554427,"5189":-0.022975938483726196,"5190":0.06026526107687305,"5191":-0.13975762536360076,"5192":0.06262739085994415,"5193":-0.09805578222795647,"5194":-0.09836360960412706,"5195":0.04245536993470642,"5196":0.15794858687551475,"5197":-0.11634072940590544,"5198":-0.024992066311225823,"5199":0.10990916858600748,"5200":-0.0456998525045519,"5201":0.1526159958184935,"5202":-0.023241432500576115,"5203":-0.018746456879264656,"5204":-0.015047854096071473,"5205":0.005062644133137667,"5206":0.06797400302020098,"5207":0.10204256042600919,"5208":-0.06672637351840643,"5209":0.15993017187361183,"5210":0.6399003429350276,"5211":0.08603857383769987,"5212":0.33932234341796186,"5213":0.21420214719203892,"5214":-0.2015317393925196,"5215":0.9756569007536902,"5216":0.8239100410022087,"5217":0.7451896844042124,"5218":0.04371730761440683,"5219":0.18199998914209778,"5220":1.1193868144275696,"5221":1.0617849862173439,"5222":-1.212804083261081,"5223":-0.3333092560611793,"5224":0.28309139327405886,"5225":-0.10168672016675127,"5226":0.6951471842242172,"5227":2.1087048054441175,"5228":-0.34328219641661245,"5229":-0.22597755541773903,"5230":0.8995610583468361,"5231":0.5669009589651984,"5232":0.17596641977717178,"5233":-0.0947859915509627,"5234":-0.6130620257627529,"5235":3.321138224507757,"5236":-0.7850146019779677,"5237":1.2062156595036744,"5238":-2.7923241993012398,"5239":0.021892726129381056,"5240":-0.02638304512364967,"5241":-0.026280558666782423,"5242":-0.0715329633147447,"5243":-0.02358751003040376,"5244":0.023074590276423636,"5245":-0.03169780169017218,"5246":0.06853591065362177,"5247":0.04405780387618677,"5248":0.08746244814033825,"5249":0.005996497721039855,"5250":-0.045631815364132945,"5251":0.0026024825259737556,"5252":0.08014598070634846,"5253":-0.02646207698650754,"5254":0.016233173800086192,"5255":0.11178185691776155,"5256":0.00868516992119528,"5257":-0.08455388297535964,"5258":-0.04454684806547519,"5259":-0.09163081322685603,"5260":-0.0531451428222111,"5261":0.05721416475416347,"5262":-0.061619762228296106,"5263":0.13529442377095258,"5264":-0.003071991089169607,"5265":-0.02087392018846037,"5266":0.010902679493756517,"5267":0.019735974748900306,"5268":-0.01651100376030878,"5269":-0.08787426324284343,"5270":-0.026665271147520875,"5271":-0.028890582159039987,"5272":-0.03249013899214183,"5273":-0.035810421304636596,"5274":0.012702032147568912,"5275":-0.012721568262414662,"5276":-0.05329717094330997,"5277":-0.009021525470678159,"5278":0.009260295155073487,"5279":0.06651568483186517,"5280":0.013479333230070204,"5281":-0.07925130503772523,"5282":0.11676826574749058,"5283":0.02024054556495956,"5284":0.020455265301988535,"5285":-0.006872752409298564,"5286":0.11373539316978283,"5287":-0.08745751970265551,"5288":-0.01775866380904346,"5289":-0.06426460473218795,"5290":0.05545018299729913,"5291":0.016447323260489855,"5292":0.08768750285318494,"5293":-0.05042011809431182,"5294":0.02881003597713502,"5295":0.04083554193481022,"5296":0.03365741754024879,"5297":0.056551489195948476,"5298":0.0769978968875964,"5299":-0.006311270428801789,"5300":-0.05748562378993861,"5301":0.02266028474137985,"5302":-0.06832318389231883,"5303":-0.09138674340755762,"5304":0.024402030496417388,"5305":-0.09658244860644144,"5306":0.027376670232343882,"5307":0.021506513196876617,"5308":0.09586469448562084,"5309":0.011332027731979194,"5310":0.08751077083237677,"5311":0.08020520668568645,"5312":-0.0429109249383187,"5313":-0.13455028229367838,"5314":0.03707319716312193,"5315":-0.14107834030881292,"5316":0.017095490664256818,"5317":-0.10137007403179887,"5318":0.06614449467843238,"5319":-0.0023392146617149953,"5320":0.08332929198251195,"5321":-0.031126560935231744,"5322":0.06609024052167813,"5323":-0.05411921524413671,"5324":0.01877803146001066,"5325":0.24614672154554162,"5326":0.005137570211890306,"5327":-0.020349278441013047,"5328":-0.044995550869756505,"5329":-0.0013041340880281577,"5330":0.009018536172676474,"5331":0.07534952644598991,"5332":0.04085784467734466,"5333":0.05030285775742399,"5334":-0.031313290963781194,"5335":0.0710146673757646,"5336":0.07901826334869419,"5337":0.004430121476483021,"5338":0.02295587182727637,"5339":-0.09572686161029924,"5340":-0.09405958991937925,"5341":-0.10624523770790667,"5342":-0.08479654856453317,"5343":0.04603117010043887,"5344":-0.06335681096567698,"5345":-0.2281155185637061,"5346":0.023276750609331227,"5347":0.0621220743240546,"5348":-0.07263244768021256,"5349":-0.1762647441796393,"5350":0.15513644804796856,"5351":0.08969351655442559,"5352":-0.02782227864394303,"5353":-0.02988549592277696,"5354":0.10116057194471036,"5355":0.06377701727091999,"5356":0.05294947121532559,"5357":-0.01557471033950394,"5358":0.04115531786639624,"5359":-0.08316533715395877,"5360":-0.0863450924815045,"5361":-0.04996234087154578,"5362":0.21240964071199014,"5363":-0.0029944681020705626,"5364":0.04687017054234506,"5365":0.005855200661830875,"5366":-0.005273572038228117,"5367":-0.013464794821232596,"5368":0.00793420143724885,"5369":0.03361205817356491,"5370":-0.052057455618984776,"5371":0.0006326897361579982,"5372":0.06549813849241648,"5373":-0.006706225741001498,"5374":0.015039917103725126,"5375":-0.0033734016825785547,"5376":0.00931087193291308,"5377":0.013041146792870457,"5378":0.017205682625365763,"5379":-0.01965253230239351,"5380":-0.07345831082928629,"5381":0.05571617234878131,"5382":0.03521555156914053,"5383":0.23964063854351908,"5384":-0.02858995705174796,"5385":0.04327639147082347,"5386":0.11791463938301712,"5387":-0.17605513274916246,"5388":0.035757697982438656,"5389":0.017834578648433942,"5390":0.0010451532090005329,"5391":-0.025347956253988623,"5392":-0.006962609038605358,"5393":0.02059698102036682,"5394":0.034748042753125034,"5395":0.03534080139109088,"5396":0.08136355403554424,"5397":-0.08651115196445416,"5398":0.12254791491234214,"5399":-0.029377412944286786,"5400":0.031243605151628048,"5401":-0.015665773881205255,"5402":-0.0389928599767347,"5403":-0.05579961514845761,"5404":-0.043565003351655855,"5405":0.11088139817539026,"5406":-0.08284653144949228,"5407":0.042714045977764,"5408":-0.013711450793187347,"5409":-0.032225436392273564,"5410":-0.005178280964141244,"5411":0.015234255862943372,"5412":0.02711666028342464,"5413":-0.04242155719890395,"5414":-0.16117790109570568,"5415":-0.020644935143214044,"5416":0.03926220733174222,"5417":-0.07633734766155692,"5418":-0.14364623183414468,"5419":-0.021600117500696178,"5420":-0.025120213426871273,"5421":0.031610555403007425,"5422":0.020234895349514354,"5423":0.0345674739444863,"5424":-0.08689384272922746,"5425":-0.25622229509229744,"5426":0.13090844334646498,"5427":-0.24723316653949481,"5428":0.10182917220740133,"5429":0.11215412665861187,"5430":0.1725197359929988,"5431":-0.2231973401037521,"5432":0.19134603742216644,"5433":0.50790802671459,"5434":0.21379292316449144,"5435":-0.2379717545539635,"5436":-0.11722716433445908,"5437":0.5658778020041597,"5438":0.7361073140413255,"5439":0.5487549214849715,"5440":-0.35295008802031314,"5441":-0.5336987511441856,"5442":0.650745802959106,"5443":0.6903035931017637,"5444":0.6212537092727701,"5445":-0.35961418299898856,"5446":-0.6410168328835122,"5447":1.546661347843121,"5448":1.1941706695699992,"5449":0.1830182600608755,"5450":0.3528805039590275,"5451":0.44158878603478163,"5452":0.044628550441706535,"5453":-0.5532850116569042,"5454":0.8575298439494521,"5455":-0.790516959659845,"5456":-0.05273958390191783,"5457":-0.0317097280678397,"5458":-0.05919508560461344,"5459":-0.0541997287278414,"5460":0.031666607437316655,"5461":0.021829222211896863,"5462":-0.010213552948763326,"5463":0.08720015447560515,"5464":0.08965659116325468,"5465":0.13345090104428745,"5466":-0.003208965182311515,"5467":-0.0635275510883324,"5468":0.04042785791234573,"5469":0.06806617663001178,"5470":-0.0027050671239420283,"5471":0.05968982647693949,"5472":0.1362167430539331,"5473":-0.003055054816914617,"5474":0.004343627420123404,"5475":0.0732002183793888,"5476":0.019506083709893526,"5477":-0.035702014241311524,"5478":-0.03045609290610056,"5479":0.12762595561502518,"5480":-0.28904846490048,"5481":0.034718453968753865,"5482":-0.007818676446577608,"5483":0.0070546976346836585,"5484":0.0019090087435552638,"5485":-0.011523202682560386,"5486":-0.10914572368029485,"5487":-0.1043508475437082,"5488":0.04430616863955024,"5489":0.005858997853477108,"5490":0.02060757465543364,"5491":0.01046080794870988,"5492":0.005608419282698698,"5493":-0.06999257128490127,"5494":0.008058356653095958,"5495":0.06263227584300332,"5496":-0.02560680006367106,"5497":0.06263948612114906,"5498":-0.05986293080602214,"5499":0.05401219046441157,"5500":0.05153656655229995,"5501":0.0038512301191963453,"5502":-0.02685042637655673,"5503":0.06130836635044203,"5504":0.0041866611858525395,"5505":-0.12612000069832177,"5506":0.03895968058406034,"5507":0.03811399012428525,"5508":-0.0520722136556501,"5509":0.05676592831695352,"5510":0.089138354705944,"5511":0.10044524297606457,"5512":0.028678374134912067,"5513":0.03101108216628207,"5514":0.028900109478673507,"5515":0.021501614932114662,"5516":-0.019168391078528507,"5517":0.01324539849843707,"5518":-0.017395193795402875,"5519":0.01273984411192285,"5520":0.05687237468335554,"5521":-0.005289101843957128,"5522":-0.02203738878076484,"5523":-0.09969476666511477,"5524":0.003487745206905829,"5525":-0.03860168288442998,"5526":-0.06931495757002971,"5527":-0.19431154977178164,"5528":-0.03914241888991248,"5529":0.08045272126281133,"5530":-0.027505500037805396,"5531":-0.11535616437343385,"5532":0.016637835974672395,"5533":-0.031667838177591875,"5534":-0.06580728886795643,"5535":-0.021959485442114866,"5536":0.1161496983946374,"5537":-0.010551553659222442,"5538":-0.1885788773310766,"5539":0.0009578401755067664,"5540":-0.1385639084659267,"5541":0.10272506956948824,"5542":-0.16886685137241134,"5543":-0.029083407843452,"5544":0.02462491975393355,"5545":0.02355552877948547,"5546":0.000907296330603304,"5547":0.03833424883149097,"5548":0.08775246370198558,"5549":-0.00466146048963904,"5550":0.04410617202508478,"5551":0.00903283008049374,"5552":0.008799899781031073,"5553":0.08036909259039281,"5554":0.03066335007996999,"5555":-0.027326806872557407,"5556":-0.01225319431068495,"5557":0.02859256160081416,"5558":-0.023824865299613376,"5559":-0.0018381562322062717,"5560":-0.051905005142584344,"5561":0.06519353574846674,"5562":-0.02933134361493316,"5563":0.11502482477330007,"5564":-0.014341579743974694,"5565":0.11415606375791819,"5566":0.011724376764082719,"5567":-0.11576439073156072,"5568":-0.0420161357721113,"5569":-0.08627560262586595,"5570":-0.017554987356750998,"5571":0.06872257144706484,"5572":-0.022774388175253322,"5573":0.05314385481036729,"5574":0.0020167590285834333,"5575":-0.0032129713299941737,"5576":0.027907129464186047,"5577":0.03798905940491805,"5578":-0.041882553281886704,"5579":-0.04866597703963648,"5580":-0.08276641977366835,"5581":-0.01611184881312344,"5582":-0.04787539269952361,"5583":0.04171981045669282,"5584":-0.06412646770701035,"5585":-0.05430474185836934,"5586":-0.05056907992987772,"5587":0.028161764155804018,"5588":0.004248233907886049,"5589":-0.1074173630320109,"5590":-0.030725971835081155,"5591":0.018665068901971067,"5592":0.09007141947576386,"5593":-0.05755245671927027,"5594":-0.07951431720120042,"5595":-0.04579838765942717,"5596":0.055238903537582235,"5597":-0.0974286963754761,"5598":0.062213156031756464,"5599":0.18139669625999097,"5600":0.08300383960230555,"5601":0.07145991015129247,"5602":-0.05413202334502243,"5603":0.09036115423663958,"5604":0.07239775791922444,"5605":0.02324242299074568,"5606":0.056756360150241096,"5607":0.01964041077358711,"5608":0.022648373450916564,"5609":-0.0845793216203227,"5610":0.202683783477796,"5611":-0.016989964764702714,"5612":0.01037620115699424,"5613":-0.03396961426818865,"5614":-0.05079684994991882,"5615":0.04740237838674756,"5616":0.03992361293206502,"5617":0.004266071773718036,"5618":0.026835122034308954,"5619":0.011495988366779201,"5620":0.06133544432712512,"5621":0.00954459530423252,"5622":0.030852542787155286,"5623":-0.07463563260826232,"5624":0.031138070590341786,"5625":-0.008923089481805842,"5626":0.03354892495400692,"5627":-0.03401886522015955,"5628":0.053214078089485466,"5629":-0.033405819783930424,"5630":0.04073599902369211,"5631":-0.27573728269297826,"5632":0.024446194658555204,"5633":-0.03800873227081406,"5634":0.00018314705182117256,"5635":-0.09248257875924791,"5636":0.0419293178692409,"5637":-0.012202169954471714,"5638":-0.03374169307422197,"5639":-0.0043482245420745505,"5640":0.016461178275856365,"5641":-0.022276946409137625,"5642":0.004025497651445447,"5643":0.03989181649294393,"5644":0.021873963315768147,"5645":0.07709413579811468,"5646":0.028769205517200348,"5647":0.032724652199808024,"5648":0.02654255568274653,"5649":-0.04710282543521583,"5650":0.013145689446715719,"5651":-0.07705143636998611,"5652":-0.02637942095898269,"5653":0.00025373187633299796,"5654":-0.04595365687125049,"5655":-0.11109143079984858,"5656":0.06956832833824098,"5657":0.03569788492741738,"5658":-0.00979220745472544,"5659":0.00040720933047894023,"5660":-0.0007030186709311576,"5661":0.008031514308750765,"5662":0.04854812488911076,"5663":-0.013646977392794598,"5664":0.014092133900734116,"5665":0.04229019886914617,"5666":0.11779243474879805,"5667":-0.022745958499390716,"5668":-0.0014822813324802601,"5669":-0.04689664650447982,"5670":-0.03131047254939041,"5671":-0.016363641505665553,"5672":0.04160033451798763,"5673":-0.02212091667724966,"5674":-0.013169497520297679,"5675":-0.004119478954577598,"5676":-0.0010527920547953843,"5677":0.040150894483928605,"5678":0.0073973515017837855,"5679":0.016000332217666006,"5680":-0.002509278482866672,"5681":-0.020619889961017438,"5682":0.0466445697803108,"5683":-0.011634447868883879,"5684":0.00241688942597721,"5685":0.017887006989477768,"5686":0.06498118279510932,"5687":-0.04692835700682955,"5688":0.04504463977280023,"5689":0.013238918463735594,"5690":-0.05359516004822279,"5691":0.019540599928634068,"5692":0.07082963694864286,"5693":-0.14302381808784537,"5694":0.0780754488887975,"5695":0.039296531644174244,"5696":-0.036272709918700026,"5697":-0.10944731923915382,"5698":0.018090324128752867,"5699":0.010344473156033146,"5700":-0.004746378127259482,"5701":-0.00007531591598832982,"5702":-0.002255658742429743,"5703":-0.012959163632132863,"5704":-0.007587572097309921,"5705":0.011863215726290904,"5706":0.03284241034121687,"5707":0.0819965179520786,"5708":-0.11223521052674007,"5709":0.014204640232215982,"5710":0.04313154774553616,"5711":0.0032732103238695227,"5712":-0.007573276729109254,"5713":0.054680546178289165,"5714":0.047195291744891864,"5715":-0.000043128052778607295,"5716":-0.017860870038687592,"5717":0.0217531251537021,"5718":-0.0457637364149251,"5719":-0.09548590945136477,"5720":-0.013412029642511893,"5721":0.06024402963188999,"5722":-0.03896547242115478,"5723":0.01146472434593128,"5724":0.4886429531954342,"5725":-0.09691969166869069,"5726":-0.05152060899020753,"5727":0.15998773884728038,"5728":0.183597104232185,"5729":-0.019399182225888018,"5730":-0.015292585646143676,"5731":-0.009782026964086405,"5732":-0.08643775380582192,"5733":-0.004066731104013378,"5734":-0.03280172832511103,"5735":0.05785466961848826,"5736":-0.08419774863980668,"5737":-0.015762749867416416,"5738":-0.008991678695061825,"5739":-0.062050390493037205,"5740":-0.059307251132037414,"5741":-0.09338495721703376,"5742":0.09511215277419595,"5743":-0.005295835816211752,"5744":-0.07856876065924533,"5745":-0.05240757783529934,"5746":-0.0036868680706326385,"5747":-0.013612394017118525,"5748":0.013808596359765533,"5749":-0.12590425635750896,"5750":-0.033105918420270504,"5751":0.113332759001004,"5752":0.09526062251648192,"5753":0.0066089216808093015,"5754":-0.006344825690365139,"5755":0.01966535882650656,"5756":-0.14287294338748976,"5757":-0.07019144552429142,"5758":-0.11508734244729739,"5759":0.09393987492553305,"5760":-0.06255011400633244,"5761":-0.06427804409291592,"5762":0.024524688893454136,"5763":0.13024834760756587,"5764":0.04999314407646789,"5765":0.025827861395321054,"5766":0.014964020959613508,"5767":-0.03240341628381972,"5768":-0.07002521135880227,"5769":0.018538921935320245,"5770":-0.01891512388626086,"5771":0.03702167284419994,"5772":0.01150294840970081,"5773":0.009866462915407999,"5774":-0.011399083910067253,"5775":0.06608917953267951,"5776":-0.00009100821763892523,"5777":-0.016798545117903654,"5778":-0.018531986485559267,"5779":-0.025721002809997826,"5780":-0.05794304971649866,"5781":0.02398661749704973,"5782":-0.026912407792864197,"5783":-0.06270571621453125,"5784":0.057066241702636834,"5785":0.052749018416096305,"5786":0.15461981559577653,"5787":0.04301259166479878,"5788":0.038992431671065565,"5789":0.026352231446078848,"5790":0.17690156348246694,"5791":0.021222535709204706,"5792":0.02272939186605839,"5793":-0.052337787174177565,"5794":-0.039217572438506355,"5795":-0.051695185990515057,"5796":0.0861989214109519,"5797":-0.05236289008471705,"5798":0.04921950123969591,"5799":0.027004869183049368,"5800":0.0015442469857811728,"5801":0.05044911009507342,"5802":-0.030030491900758457,"5803":-0.04974655547893821,"5804":0.011883402213291625,"5805":0.054111421835445726,"5806":-0.026769809576610456,"5807":0.024337924117637428,"5808":-0.013984779078058748,"5809":0.03623221751238374,"5810":0.04290260492429468,"5811":0.02871278103840961,"5812":-0.0013080433623973455,"5813":0.09828326711115279,"5814":-0.008970363389196889,"5815":-0.049883317870234686,"5816":0.03816550191078712,"5817":0.0014333802171571147,"5818":-0.10549264322632868,"5819":0.05517226731510226,"5820":0.038975553129550526,"5821":-0.13463951199453245,"5822":0.009989002285252695,"5823":0.005976470455152208,"5824":0.04861050440113897,"5825":0.03778362041334662,"5826":0.02614560204839206,"5827":-0.048494971618132404,"5828":-0.021984434346235947,"5829":0.005564149764718985,"5830":0.024104228781695144,"5831":0.15854369533379717,"5832":-0.015452054030820934,"5833":0.005914230455005997,"5834":0.0689296841768074,"5835":-0.06357541152730013,"5836":-0.08534239381876044,"5837":-0.0942792997457854,"5838":-0.03215515259316817,"5839":0.07533617672231474,"5840":-0.049320475825610874,"5841":-0.1376412967113231,"5842":-0.05027575628003349,"5843":-0.016128620720743538,"5844":-0.129368298427869,"5845":-0.02358131060282126,"5846":0.1370757928371851,"5847":0.02896979633700768,"5848":0.022678143878838448,"5849":0.1453690251458958,"5850":-0.017111594962800465,"5851":0.03256690588282978,"5852":-0.0666739380756004,"5853":-0.01760232190670889,"5854":0.04349992012271679,"5855":-0.05385261561327304,"5856":-0.08887630343970152,"5857":-0.026959594612483213,"5858":0.11017873600806553,"5859":-0.04732715593922526,"5860":0.040419346015070204,"5861":-0.016467692762630035,"5862":-0.00974657524884499,"5863":-0.17860400847315142,"5864":-0.07467435229990732,"5865":0.05681337983312332,"5866":-0.040709878430138464,"5867":-0.010915790269170708,"5868":0.061230696255941686,"5869":-0.04554358017642785,"5870":0.09946269867817042,"5871":-0.12919911506732293,"5872":-0.007849126566439381,"5873":-0.12417700865805816,"5874":0.08146277425357967,"5875":-0.010061608207850247,"5876":-0.012826135436207954,"5877":0.10120736325021154,"5878":0.13242339172689227,"5879":-0.4640188785141936,"5880":-0.13951591110725872,"5881":-0.12411548061901943,"5882":0.2461017940118308,"5883":-0.29522637817047687,"5884":-0.07838038847014783,"5885":-0.03985138684751403,"5886":-0.04152937182951863,"5887":0.05588348111763524,"5888":-0.05975649486548233,"5889":0.09372612757451362,"5890":-0.013117286786095925,"5891":0.044935163217100144,"5892":0.05929254700980358,"5893":0.023925706803134734,"5894":0.13029910778480686,"5895":-0.028189679365904316,"5896":-0.0723448584710666,"5897":-0.06332681129796959,"5898":-0.01612863924821358,"5899":-0.171816317873728,"5900":0.014447433175946583,"5901":-0.0027259360000829293,"5902":0.08119011429177941,"5903":0.03044460373158795,"5904":-0.03732421539107152,"5905":-0.0007429544273494565,"5906":-0.03761298257832305,"5907":-0.048902413056361525,"5908":-0.05182961066898527,"5909":0.023546885225224627,"5910":0.08343846199118671,"5911":-0.043288328468418846,"5912":0.04876356512785051,"5913":0.002047993339483768,"5914":0.12731002218475534,"5915":-0.002245661605111906,"5916":0.008985280367526213,"5917":0.030501354667260037,"5918":0.024148098682196358,"5919":0.04766648478270259,"5920":0.04719897282381033,"5921":0.030757732945220547,"5922":-0.010626591305515437,"5923":-0.018725151059061165,"5924":0.01947585162895926,"5925":0.021634953518540764,"5926":0.0412906923821194,"5927":0.03751646829901614,"5928":-0.05801760610380785,"5929":-0.09745001760403198,"5930":0.0004039832896638415,"5931":-0.03930720826916633,"5932":-0.0023173539826069293,"5933":-0.022949767652625692,"5934":-0.09843706259480271,"5935":0.06755912372329011,"5936":0.02337404423573942,"5937":-0.04348307522563786,"5938":-0.006594773051364441,"5939":0.04336967408626292,"5940":-0.1385470349283685,"5941":0.13508944093310685,"5942":0.03661239003488929,"5943":0.010546416425970005,"5944":0.03605695094044314,"5945":-0.04075424807713619,"5946":-0.03546128568957142,"5947":-0.0008363329733399373,"5948":-0.03020825163257948,"5949":-0.07251796320326889,"5950":-0.011606547789061502,"5951":-0.020720058721217592,"5952":-0.010057328145821455,"5953":0.021523885499393165,"5954":0.02284852638090435,"5955":0.059111110661392,"5956":0.028195481127885728,"5957":-0.03407733771356392,"5958":0.041193123939326975,"5959":-0.04228525719342938,"5960":-0.09406567616654622,"5961":0.02216290984544291,"5962":0.036762323342214985,"5963":0.04050219250921456,"5964":-0.01757311152365594,"5965":-0.010652238908619712,"5966":-0.025979098365233995,"5967":-0.028080670145363678,"5968":-0.11683649893196847,"5969":-0.03862648454441832,"5970":0.039718842761488383,"5971":0.02806164164600323,"5972":-0.006744698060967328,"5973":0.020532053521310545,"5974":0.05617481530790318,"5975":0.10232488247197445,"5976":0.1059250507386304,"5977":0.012537131937525042,"5978":0.032526042040008196,"5979":-0.005368944912516263,"5980":-0.00796362703322335,"5981":-0.013783924997249753,"5982":0.08841201420670258,"5983":0.06730033065526164,"5984":-0.035929241296442156,"5985":-0.018027309498654425,"5986":-0.016724430828717638,"5987":-0.02904666439773173,"5988":0.021138094763102716,"5989":0.04238812896726527,"5990":-0.014465175604868671,"5991":-0.10789531379601057,"5992":0.18798665498254866,"5993":-0.04291926583358358,"5994":0.046614322442414965,"5995":-0.07824167166750572,"5996":-0.007292309096952223,"5997":-0.015492153946519588,"5998":0.004396030133799342,"5999":-0.018819889413654544,"6000":-0.030935965238537897,"6001":0.09142257588468317,"6002":-0.15811959170594925,"6003":0.029872647231932108,"6004":0.011215095369502064,"6005":0.06075665897163394,"6006":-0.003265966576174356,"6007":0.0735316285303724,"6008":-0.027581176909450708,"6009":-0.005598975392580888,"6010":-0.043438240579057215,"6011":-0.023133991583447075,"6012":-0.027762600284722803,"6013":-0.046887172174645594,"6014":-0.0393119174226125,"6015":0.046472673181994814,"6016":-0.02610678809534886,"6017":0.015779056278774618,"6018":-0.03563912864738708,"6019":-0.013128132751126408,"6020":0.034404862815400644,"6021":-0.009739316052830545,"6022":0.013546069285690977,"6023":-0.006662702087319579,"6024":-0.010535612604070856,"6025":0.037148510660281464,"6026":-0.023896292258572434,"6027":-0.09355896005206535,"6028":0.0575942061552965,"6029":-0.03247627244942768,"6030":0.017840210610607796,"6031":0.04187033093382261,"6032":-0.018062186509819358,"6033":0.019823891244173424,"6034":-0.008258719916049874,"6035":-0.01749728802048032,"6036":-0.06032726302087581,"6037":0.08269194052142369,"6038":0.10753153055019483,"6039":0.005737335403546567,"6040":0.012891492791017742,"6041":-0.009526784927109856,"6042":-0.014869243028969354,"6043":-0.039372380095808034,"6044":0.057168453333873034,"6045":-0.009473819493479273,"6046":0.012573707632704389,"6047":0.06636597243043185,"6048":-0.09702499547532932,"6049":0.07009432256436109,"6050":-0.06571793551873027,"6051":-0.009289738227250356,"6052":0.09138792523220561,"6053":0.10274212320023393,"6054":0.018518377881080846,"6055":0.04781115498648598,"6056":0.024932652004447504,"6057":0.06291372867328217,"6058":0.136162215507612,"6059":-0.008788378355112553,"6060":-0.04532805120745825,"6061":0.016650674374410902,"6062":0.0020652795138567676,"6063":-0.06638747393312405,"6064":0.017501753832466396,"6065":-0.17395596074227862,"6066":0.002525527931327712,"6067":0.10970704280222261,"6068":-0.2715893598198336,"6069":-0.09264254605970303,"6070":0.023181341019187972,"6071":-0.032903906794044396,"6072":0.10158081883669946,"6073":0.06743544911708602,"6074":0.0017532997665072993,"6075":-0.1598779078781915,"6076":0.006937032503279586,"6077":0.09812554576933173,"6078":0.09215374437213489,"6079":0.010333492620903603,"6080":-0.06803940126342045,"6081":-0.007776043040300645,"6082":0.026035704545537042,"6083":-0.02140992632654544,"6084":0.018767937033099574,"6085":0.033887498103170925,"6086":0.014473893980848659,"6087":0.024509328886344245,"6088":-0.048121491924356226,"6089":0.07071896418234692,"6090":-0.0026740834420559373,"6091":0.032467092452317166,"6092":0.005729003792701124,"6093":0.048563843915597676,"6094":-0.03241801944466555,"6095":-0.1096001001440217,"6096":-0.05106767768671106,"6097":-0.04412357743629888,"6098":-0.008295673359598726,"6099":0.08217404745350054,"6100":-0.06513503179293702,"6101":-0.021857533706459646,"6102":-0.0198980229544067,"6103":0.015950963875815724,"6104":0.0032454464774446293,"6105":0.025409336659435858,"6106":-0.08437149611220231,"6107":0.009521969541917004,"6108":-0.02514794067024041,"6109":0.017824597125012265,"6110":-0.037577143189705536,"6111":-0.10372508481086115,"6112":-0.031064531935107404,"6113":-0.06635158028747302,"6114":-0.03040933670171174,"6115":-0.062010543569720196,"6116":0.13388407596305638,"6117":0.0313024701458068,"6118":-0.01851472506625846,"6119":0.040214529026696776,"6120":0.11700030436601573,"6121":-0.08972137105862359,"6122":-0.02660873360632324,"6123":0.007532037015070221,"6124":-0.06667782896405895,"6125":0.0681041466323443,"6126":0.08459523028595133,"6127":0.26096521826746716,"6128":-0.07840823584068603,"6129":0.059309539399055505,"6130":0.0633453475517567,"6131":-0.016687610974750418,"6132":0.007750213255299031,"6133":0.005665811590942819,"6134":-0.011345913831934265,"6135":0.0010133664051543932,"6136":-0.002426155326913425,"6137":0.049517568799453585,"6138":0.056687136378051985,"6139":-0.03282526998715,"6140":0.05775363230927867,"6141":-0.08947648860159428,"6142":0.1102921463595459,"6143":-0.023763506277626547,"6144":-0.012001440979502957,"6145":0.0012262148608026228,"6146":0.04335499732558834,"6147":0.038981624756280996,"6148":0.04676819597324886,"6149":-0.002668028582996752,"6150":0.0254633254215274,"6151":0.14685459560735087,"6152":0.04582742431138595,"6153":-0.07399812201665253,"6154":0.0873220513296278,"6155":0.09200482128738866,"6156":-0.16229343969473028,"6157":-0.14596389102865404,"6158":-0.043417782564994054,"6159":-0.09611294378579681,"6160":0.04484317251626232,"6161":-0.10420046111672215,"6162":0.003948013385316482,"6163":-0.039298927800409146,"6164":-0.07660269014274869,"6165":0.08756512035191294,"6166":0.0604169857114305,"6167":0.08065813504364772,"6168":-0.26831119198084985,"6169":0.0379760564023889,"6170":-0.0678612637642055,"6171":-0.11102016140015634,"6172":0.14466937117524192,"6173":-0.12988376341708308,"6174":0.10660508219371156,"6175":0.023815120233801862,"6176":0.03443675300683544,"6177":-0.05774267211471602,"6178":-0.02717995853144732,"6179":0.05095288297875495,"6180":0.025847743172474517,"6181":-0.04466295593829545,"6182":-0.130476598535284,"6183":-0.06529531095948818,"6184":0.06237350957070296,"6185":-0.10011546802003496,"6186":-0.0521871979896428,"6187":0.16747031402500054,"6188":0.1682750832729751,"6189":0.2084756512146342,"6190":-0.05821801363965851,"6191":-0.18920346978285782,"6192":0.20294847517314557,"6193":0.026350861931783782,"6194":0.022213558950662484,"6195":0.106375557201405,"6196":-0.2678050528873912,"6197":-0.036746917509494596,"6198":-0.20086406266022186,"6199":0.436192865602102}},"b1":{"n":200,"d":1,"w":{"0":0.05447305274637384,"1":-0.017700469931634457,"2":-0.07611796728727033,"3":-0.014396920275933114,"4":0.02748657926019555,"5":-0.00030613720331274224,"6":0.07114212075704178,"7":0.02162234432929857,"8":-0.06051668520785468,"9":0.008535073563863491,"10":0.016822257451354728,"11":0.17113084789611777,"12":0.038790030310962544,"13":0.03768371237578538,"14":-0.022719972989715932,"15":0.004724132438758933,"16":-0.01667691736507849,"17":-0.05462360679085965,"18":0.03861278047821994,"19":0.15695642513555588,"20":0.05807669484355899,"21":-0.0210349186610861,"22":-0.03667729076901393,"23":0.016302795286098896,"24":-0.812977392502658,"25":7.853682366624317,"26":0.019852744584144693,"27":-0.26864117306680246,"28":0.02169955357529842,"29":-0.06629049622003862,"30":-0.02742169386962349,"31":-0.08640326309869613,"32":-0.04464010094828705,"33":-0.000347880513519553,"34":0.022038186798075926,"35":-0.005879651934464251,"36":-0.08142280334550851,"37":-0.07180654488094806,"38":0.07629619869231848,"39":0.04024543871902172,"40":0.04519786746476515,"41":0.04537501386486161,"42":0.07726962806898514,"43":0.01865778376597856,"44":-0.005530789130770582,"45":-0.051249737552348996,"46":0.003653753610636258,"47":-0.007055625374893844,"48":-0.001124376627412688,"49":-0.04214213389278613,"50":0.04393721778264984,"51":-0.0815995957390897,"52":-0.05132794921309844,"53":-0.03860788865883016,"54":0.0510408327048884,"55":-0.1422873451674019,"56":0.016016887089347467,"57":0.013244092461877358,"58":0.03299782390584286,"59":-0.007943870831627927,"60":0.09439144198000662,"61":0.037562292543446354,"62":-0.1856181429692484,"63":-0.07455928852584444,"64":-0.06487651285392333,"65":-0.01581334491956155,"66":-0.004859301470609985,"67":0.0028666998341235327,"68":0.03332183103836768,"69":-0.015281475396566612,"70":-0.0653202220802484,"71":0.07802568897224774,"72":2.712185767820286,"73":-0.04270216614192011,"74":-0.11139591094004836,"75":-0.06006248583990569,"76":-0.0005595983907722833,"77":-0.027615413984015538,"78":0.010482131979451683,"79":0.04454746612811789,"80":0.055693017649463133,"81":0.017819976285187257,"82":-0.013903376409543831,"83":0.022480271803797035,"84":-0.0294249877054753,"85":0.08033684176039466,"86":0.06996934291627592,"87":-0.2547161806318922,"88":0.026002967458245554,"89":-0.08973075279068443,"90":-0.0069540917751318865,"91":-0.055772375931551245,"92":0.012744420159809268,"93":-0.03609257812719612,"94":-0.04602396828464545,"95":-0.027223000226149353,"96":0.0057662220499391246,"97":0.06266376531719492,"98":0.03352190504346188,"99":-0.06856517585199674,"100":-0.033870701768602116,"101":-0.04826487791844299,"102":-0.08086554790896602,"103":-0.37706316249056626,"104":0.0801393564558711,"105":-1.8700133429782146,"106":-0.07476725037576064,"107":0.04688204075334506,"108":-0.017913122333489424,"109":-0.025957164127482397,"110":-0.4052807563806248,"111":-0.12379935158112965,"112":0.10768215515165526,"113":0.06550876349356431,"114":-0.06378523434579758,"115":-0.013948627753352495,"116":-0.03343162121401652,"117":-0.026422075319243135,"118":0.03400932661465965,"119":0.004364241494456866,"120":0.0723476383020478,"121":-0.10746957622271708,"122":-0.04662483887130244,"123":-0.013343856410656225,"124":-0.02305292438702526,"125":-0.011118933930409617,"126":0.18776154546921367,"127":-0.10629057442629292,"128":0.017507867714196632,"129":-0.04863674532049661,"130":0.018624071852130258,"131":0.09622268868781854,"132":0.004189131269139552,"133":0.13899376327730106,"134":-1.3940192798088447,"135":0.02947695176631567,"136":-0.011222530002773436,"137":-0.029690186594235606,"138":0.07083440307490338,"139":-0.20712720136627058,"140":0.019986840740599924,"141":-0.005592130084312193,"142":0.1746218696449778,"143":0.02703473033508156,"144":-0.02918126219755671,"145":0.043442926830992414,"146":0.015959680912827734,"147":-1.9403677598911322,"148":0.009239383731232268,"149":0.006981836052721135,"150":-0.022161058993405678,"151":-0.01652390081335657,"152":-0.011154705284925553,"153":-0.009149788892225471,"154":-0.018505432778736956,"155":-0.043821536791018675,"156":0.03376663636165391,"157":-0.04411673467877406,"158":-0.026157788090546544,"159":0.02635710616049378,"160":0.02509577856264378,"161":-0.07326697681058188,"162":-0.009787561073376028,"163":-0.009675803958818788,"164":-0.002329217033962344,"165":-0.079336535482546,"166":-0.009162101324881891,"167":-0.0423190049473525,"168":-0.40614601881856627,"169":0.051043879190515944,"170":-0.09391003515081431,"171":0.03287131281623439,"172":0.020841448092340596,"173":-0.057270456063447356,"174":-0.008645260568513578,"175":-0.40119000950273387,"176":-0.020570068757509438,"177":-0.05253840869245513,"178":-0.08256354965026978,"179":-0.015662992932240553,"180":-0.0758594959354901,"181":-0.007125667867404731,"182":0.05100425058143193,"183":-0.0398731964168273,"184":0.09631679297111756,"185":0.03591177918778386,"186":0.05171729756701637,"187":-0.08225631199088199,"188":0.04381971608689538,"189":-0.07293159111761288,"190":-0.03151287955377571,"191":0.11162744595581721,"192":-0.04995432766913529,"193":0.09173747221993177,"194":0.008495912536600588,"195":-0.04147361109082109,"196":-0.016463530683096604,"197":-0.01698045312674468,"198":0.05691483913670053,"199":0.050635690008932775}},"W2":{"n":5,"d":200,"w":{"0":0.03416852189229679,"1":0.08654339558783469,"2":-0.05456731219312468,"3":-0.008511851787591644,"4":-0.011118668844321906,"5":0.029290911104832375,"6":0.05322209338495907,"7":0.014269541964713897,"8":-0.009818543052123516,"9":0.004773524717429415,"10":0.023387041368550434,"11":0.0892016646954704,"12":0.10283476834976386,"13":-0.062261947491098535,"14":0.035092214277781864,"15":-0.0011189407862882625,"16":0.043010229169731955,"17":0.056492612699437325,"18":-0.02202217069228083,"19":-0.14312641971610518,"20":-0.003268675838561939,"21":-0.03411152298689757,"22":0.04133687537530974,"23":-0.007029968794800561,"24":0.3387535896235944,"25":-5.606507537547282,"26":-0.03093455503537644,"27":0.9479541350183158,"28":0.0031876092442935204,"29":-0.02149877675332721,"30":-0.08896400664429101,"31":-0.07693923299305017,"32":0.08787516621717506,"33":0.010822434906244266,"34":0.02731129932255105,"35":-0.016742317934334016,"36":-0.039065416677929024,"37":0.1068392888215313,"38":0.06296999059951747,"39":0.05910298410880956,"40":0.01530766067338707,"41":0.02987182808252181,"42":0.07273019821273995,"43":-0.09293104513573124,"44":-0.06975814577549716,"45":-0.014121079614768032,"46":0.006503319396023746,"47":0.046932720900596143,"48":0.024943022665722254,"49":0.03452749486255925,"50":0.09275133957706445,"51":-0.16980385698238362,"52":0.052401295858597145,"53":-0.05815660289822201,"54":0.003779221697562229,"55":0.10588811880276332,"56":0.04045526604841512,"57":0.02311336737444661,"58":-0.06745662857026653,"59":-0.022615586069732328,"60":0.009369679692825315,"61":0.053871819375527055,"62":0.41833877182592033,"63":-0.006548946351100279,"64":-0.004738956894642354,"65":0.0484899310757978,"66":0.05057465393193275,"67":-0.01864003036574894,"68":-0.04528689375236503,"69":-0.0030210015061036166,"70":0.009285431451438998,"71":0.020886994946127387,"72":1.501842654045354,"73":-0.024802658187165226,"74":-0.04695991055826394,"75":0.03360643491237327,"76":0.07141604070009142,"77":-0.01823195252359464,"78":0.025190953508246284,"79":-0.023509416783956412,"80":0.04098072494678349,"81":0.05711206953020661,"82":0.09594907478383596,"83":0.043845173351746335,"84":0.002797944083934391,"85":-0.007468579335459617,"86":-0.02593137272825763,"87":-0.2006117844116143,"88":0.0054319394385459385,"89":0.0073797344552251395,"90":-0.003645219681271975,"91":-0.04454604702995835,"92":-0.03762493741250984,"93":0.05830392490644361,"94":0.023677112845552597,"95":0.04568772674775856,"96":0.04329761337317285,"97":0.04984102274483556,"98":0.05171806374228654,"99":0.008664029320728205,"100":-0.05218568519079394,"101":-0.011051972247185198,"102":0.0013695164513146342,"103":0.10272947962375033,"104":0.026511478029013005,"105":4.502372972518313,"106":-0.1605098306414938,"107":0.05682989782671301,"108":0.06829842150213006,"109":-0.0820399718015332,"110":-0.21085981037217574,"111":-0.07034026322994903,"112":-0.07282988184797451,"113":0.007666137483490569,"114":-0.046424996381378,"115":0.04583496606589236,"116":-0.04746371523288278,"117":-0.023229844758318696,"118":0.04485008809792044,"119":0.024757936177559583,"120":-0.054231117282718594,"121":0.16075775492767794,"122":-0.05342694830201469,"123":0.06724660016482226,"124":0.025244582229453857,"125":0.02148215094006986,"126":0.06251641576872695,"127":-0.038361259925400494,"128":-0.014831600303321858,"129":0.04923452319796648,"130":-0.028502465820319782,"131":0.021336189459429604,"132":-0.046796430574306075,"133":0.09498644165457013,"134":-1.2439570934618858,"135":-0.006442011800611089,"136":-0.06504983548751803,"137":-0.012000436843744923,"138":-0.030005635305185425,"139":-0.6123110956901701,"140":0.010334108950141708,"141":-0.0403453946888686,"142":0.13286416888788655,"143":0.011800288480226443,"144":-0.005151999478452622,"145":-0.01289974271776955,"146":-0.0054615642772869975,"147":-4.547040427057905,"148":0.055133434408289855,"149":0.052885296976595095,"150":0.05356214579554277,"151":0.03737239546213567,"152":0.0017467673132967481,"153":0.012040178820480345,"154":-0.008234264929778915,"155":0.03856027686733882,"156":-0.025960634715093335,"157":0.0787614894559476,"158":0.04981395831749456,"159":-0.011034059120830305,"160":0.06055001630581311,"161":-0.03528897108503373,"162":-0.03519038583283515,"163":0.028163052957520113,"164":-0.012888467516842785,"165":0.04730509611672063,"166":0.02235638098233982,"167":0.09411521100153705,"168":-0.0925470898679314,"169":0.028062237221378003,"170":-0.005452334714576562,"171":0.017212724572045878,"172":0.1302528841865132,"173":-0.03687845815624714,"174":-0.02161115634458839,"175":0.6411052311959002,"176":-0.0322120618193312,"177":0.03912939478284201,"178":-0.07345433127123278,"179":0.04124158849725957,"180":0.03780549906418436,"181":-0.008136072394090196,"182":0.06385115776237361,"183":0.00909337678148028,"184":0.0025184113083274924,"185":0.03876287351246745,"186":0.06565711846027014,"187":-0.0028884998555306477,"188":0.022443435181556098,"189":-0.0905002340467345,"190":0.07124280194977621,"191":0.009278542467620686,"192":0.020168236747514785,"193":0.009000703449375877,"194":0.013058256810756997,"195":-0.05924528723067463,"196":-0.05494646887324114,"197":-0.004123111710927593,"198":-0.03836505318048393,"199":0.09333047949682398,"200":0.029082320094184492,"201":0.01986971497354653,"202":-0.0014834631987330111,"203":0.004949862702902208,"204":-0.010590355792490411,"205":0.04393236532538431,"206":0.03703889748480236,"207":-0.05147788081526613,"208":0.033307886960233485,"209":0.010875159541110185,"210":-0.0828956958389561,"211":0.03161655125037346,"212":-0.021172647019284202,"213":-0.02381170249952363,"214":0.06100309456307708,"215":0.032646888803866375,"216":-0.0004199302883743381,"217":-0.005987940055637902,"218":-0.015399149269877063,"219":-0.3125559690737756,"220":0.030888821367583936,"221":0.0712234087165018,"222":0.032977194242711415,"223":-0.057088507922365556,"224":0.6541997813195994,"225":-4.129756516048222,"226":0.010780404509603216,"227":1.2484650981828542,"228":0.09159613683331849,"229":0.030423652943495288,"230":-0.02580895226808874,"231":0.08502512882567184,"232":0.03225742435645076,"233":-0.013991385728890856,"234":-0.011606538716740476,"235":-0.03424859344913859,"236":-0.001316907031668267,"237":-0.09349043695462954,"238":0.028282695144058293,"239":0.10677699491472216,"240":-0.051956482823293224,"241":-0.010743661391664926,"242":0.007646582577520499,"243":0.010860040260557592,"244":0.11270176231353803,"245":-0.07994238785639425,"246":0.0895497260106574,"247":0.033942877753640535,"248":-0.017058602593343117,"249":-0.1299657779312692,"250":0.10388388477427288,"251":0.014548757879950244,"252":-0.004158070661062461,"253":0.013636659980809861,"254":0.013090646743258393,"255":0.006108393935851458,"256":0.11440300743711812,"257":0.002386373439858505,"258":-0.016657216310284256,"259":0.02802879336317639,"260":0.014910330793793908,"261":0.007168436954825971,"262":0.012874773546956276,"263":0.004596947242914014,"264":-0.05110111588420007,"265":-0.04901499607018782,"266":-0.015775355646488236,"267":0.00005090375344787716,"268":0.06049626696221813,"269":0.06792016745079464,"270":-0.061217895608455836,"271":-0.005719885662852968,"272":1.859623994240935,"273":0.026523470130453575,"274":0.010557298058922773,"275":-0.027291537823974883,"276":0.035751691320170365,"277":-0.04865233748700084,"278":-0.034073304269839774,"279":0.001291021399305897,"280":-0.0413314825469192,"281":0.015255850485989956,"282":-0.03386765489501546,"283":0.007892892044631651,"284":-0.0552069749729618,"285":0.0010165767850948302,"286":0.08969824717992847,"287":0.062487987575253326,"288":0.03358930842637017,"289":-0.09763381439576348,"290":-0.05364033164962824,"291":0.012940992053557551,"292":-0.010634536490814762,"293":0.008271458049160437,"294":0.009835024135990007,"295":-0.022589653826798325,"296":-0.0827205021585404,"297":0.04791759041052823,"298":-0.03630685870971956,"299":-0.01749598422873298,"300":-0.03176106517400409,"301":0.04596430977530167,"302":-0.044293589250678805,"303":1.232254248113634,"304":0.08211087530517895,"305":4.035421715903115,"306":-0.07518375134619175,"307":0.0373128977489899,"308":0.05176749681332626,"309":-0.05321919230803938,"310":-0.3261121091707228,"311":0.007073313455391271,"312":-0.19188076911863106,"313":0.12471988741417828,"314":-0.017212900507377605,"315":0.018818837786786273,"316":-0.0026590527642749787,"317":-0.09083950275723703,"318":-0.0635313720530816,"319":0.040361195900519106,"320":0.08765397648095324,"321":0.12583297170938582,"322":0.016956115232668136,"323":-0.03057570735638334,"324":-0.06859522322002076,"325":-0.05127976896956138,"326":0.0812162646809538,"327":-0.09005002772204104,"328":-0.043376114121382224,"329":-0.03766980668551685,"330":-0.0025219698416647828,"331":0.05626913308772027,"332":0.019397834803032737,"333":0.05226804097185439,"334":-1.6429900091311282,"335":-0.017223715638123617,"336":-0.03389960996150775,"337":-0.07319739818369168,"338":-0.017848060785819235,"339":-0.6804755870857115,"340":-0.0019820251530078443,"341":0.08179581410048589,"342":-0.026570811982951165,"343":-0.037945406812483304,"344":-0.055875602897540816,"345":-0.05411643709073156,"346":-0.006586302405411507,"347":-4.0956246637129485,"348":-0.1391585088317921,"349":0.012065461932258553,"350":-0.012827066787658,"351":0.03519301462123561,"352":-0.038079587664376045,"353":0.01999012291065339,"354":-0.09996319242178764,"355":-0.011179564679423581,"356":0.03411144028898481,"357":0.02810528300557658,"358":-0.0005975448980677015,"359":0.023238739600102712,"360":0.0477176361430436,"361":0.03287826095283709,"362":-0.06390028492087549,"363":0.028307170041552723,"364":-0.09860578806320786,"365":0.03585255346843633,"366":-0.08363622379445976,"367":-0.020699055576658014,"368":-0.04916200930866008,"369":-0.062018899403877635,"370":-0.027453664324108472,"371":-0.02807197416246248,"372":0.08320866418170425,"373":0.011108854640720471,"374":-0.00012634867789249626,"375":0.798202181542171,"376":-0.03313610073318949,"377":-0.03643964897796963,"378":0.0637645370329748,"379":-0.021810163456929366,"380":0.06702226816187153,"381":-0.07403249950553599,"382":0.03640509964112371,"383":-0.035205588700757066,"384":0.03377204239567938,"385":0.07794855316652478,"386":0.006619264668003966,"387":-0.009746908378376107,"388":0.10027600956726844,"389":-0.0653004812242239,"390":0.02368731016698723,"391":0.06971507746851989,"392":-0.02247860237525011,"393":-0.010588269386045497,"394":-0.007931204628592571,"395":-0.06461258872506996,"396":-0.024229113052135095,"397":-0.016129369015212634,"398":-0.07975476087247502,"399":0.16680491360992178,"400":-0.16538407921501497,"401":0.08583063165683473,"402":0.040680707883646514,"403":0.03615154865529131,"404":0.008716818298049238,"405":-0.030086496952094803,"406":-0.0488845110732225,"407":0.0003538839348049199,"408":0.08045385169127765,"409":0.035554245804523946,"410":-0.12183348421064762,"411":-0.12344001480513418,"412":-0.19642788497606797,"413":-0.11591963769599664,"414":0.023807618023559636,"415":-0.11043925400466612,"416":0.09848455587107102,"417":0.0711097364938703,"418":0.021184857274890424,"419":-0.06031156464317384,"420":0.02043805968738645,"421":-0.11354447367925694,"422":0.06499618438929593,"423":-0.012397035670338613,"424":0.855028402723723,"425":-2.5728464038360106,"426":-0.06739502640233377,"427":1.2248433268157526,"428":-0.10414033916605486,"429":-0.12648516813063204,"430":-0.03796179605971814,"431":0.08262578785892694,"432":0.04127798621322868,"433":-0.05530390873346514,"434":-0.09374579111305915,"435":0.14184920579831908,"436":0.09662700559152809,"437":0.02330965715744883,"438":0.04657955082528614,"439":-0.09764909213211467,"440":-0.047252060610926086,"441":0.07660307683584451,"442":-0.17967039198089946,"443":-0.01563053100460093,"444":0.05073636358597205,"445":0.11562327946752546,"446":0.013699893520427246,"447":0.0048015658382467045,"448":-0.006272491214411221,"449":-0.0038426838860457636,"450":-0.015482406425956002,"451":0.013238641554922433,"452":0.0031804423111643863,"453":0.04438648552878345,"454":-0.04036882628142868,"455":-0.008122979201158013,"456":0.03305414925182106,"457":0.04452410563920841,"458":0.0580866770655777,"459":0.051105171889076276,"460":-0.10047086008511903,"461":-0.027111807244049973,"462":-0.15102723752564962,"463":0.12710910997527117,"464":-0.027569324391068885,"465":0.10491427343318661,"466":-0.033218250117547615,"467":-0.0180151426593019,"468":-0.02719696481232929,"469":-0.12430116618030482,"470":0.12388034824276786,"471":-0.095615065982217,"472":1.458725657848482,"473":0.0044641387200681035,"474":0.20148638147836703,"475":-0.04309979841422839,"476":0.11830905416286819,"477":0.01753162872024216,"478":-0.023972022927175343,"479":-0.06574174454516263,"480":-0.12867512122144606,"481":-0.027964124738489082,"482":-0.003716977239993107,"483":-0.061084991769354106,"484":0.04404792676989006,"485":0.0024731418897743513,"486":-0.031313974756347826,"487":-0.23222233561688557,"488":0.1388976593967684,"489":-0.10330397649624612,"490":-0.11456008452517064,"491":0.10511876057757152,"492":-0.09708473544669735,"493":0.01475821145877005,"494":-0.05550033956760075,"495":0.02879096146066733,"496":-0.057145389999029156,"497":-0.0864533777565417,"498":-0.16451751635230094,"499":-0.015607267187136871,"500":0.10500264570739232,"501":-0.028269195364519203,"502":-0.004227628058326277,"503":1.1383670961272205,"504":-0.07091407515983521,"505":3.9895035455873513,"506":-0.08596223207395165,"507":-0.0876171862150353,"508":-0.11180428903132561,"509":0.006826471984669912,"510":-0.47319222183852644,"511":0.09107817495492353,"512":0.05167699683761812,"513":-0.1412066345209871,"514":0.013872896318370305,"515":0.020603704825690197,"516":0.0004136066767449866,"517":-0.035519618604402996,"518":0.18292247951174057,"519":-0.034063978586895624,"520":-0.1236394152213337,"521":-0.05890535807182718,"522":-0.02722413338791374,"523":-0.03397485802353632,"524":0.03755868831559203,"525":0.14922073621969162,"526":-0.06738793828550618,"527":0.015094227581288106,"528":0.11418439461823617,"529":-0.021018747404421157,"530":-0.1862789605177922,"531":-0.1401260328612093,"532":-0.031079522632812457,"533":-0.008086247785745697,"534":-1.482178057346655,"535":0.06962367167633474,"536":-0.013869761424224056,"537":0.11620923783692676,"538":0.09996556918856783,"539":-0.48311140949149,"540":-0.01072096586494905,"541":-0.09712606489441487,"542":-0.0993925341470531,"543":0.02422985293221732,"544":0.18108144992000266,"545":0.019962466335304033,"546":0.15962553622760753,"547":-4.060659789239785,"548":0.07882550594472103,"549":0.062159804490567024,"550":-0.11002043414868777,"551":0.05989135777348822,"552":-0.0908807718598187,"553":-0.013425028855190169,"554":0.047539649805694376,"555":0.04560553450562033,"556":0.0012314449161701696,"557":0.06986941845173424,"558":-0.05137092671093704,"559":-0.10562270853129775,"560":0.034956924141555955,"561":0.03543379362239066,"562":0.08318843834703428,"563":-0.07157735512105055,"564":-0.0381667517549738,"565":0.0427630179201475,"566":-0.017016690096431666,"567":0.04554437071106693,"568":-0.08021870472138641,"569":0.02877610661020175,"570":-0.06011109607342428,"571":-0.013533429904637033,"572":0.046638082341989406,"573":-0.11456983229393786,"574":0.09210518209227331,"575":0.7009361427611678,"576":-0.010115024420485438,"577":-0.02606704193022194,"578":0.15457971946643204,"579":0.040496447589004225,"580":-0.0446109675143342,"581":0.12710968105468679,"582":0.0030386014006142836,"583":0.040579240299369955,"584":-0.2016917948839882,"585":0.037695731927072755,"586":-0.10313434160514615,"587":0.022891551244490865,"588":0.00266928914479156,"589":0.3301104669933262,"590":-0.04674222976132667,"591":-0.05916552315068416,"592":-0.01361861349928808,"593":-0.029896093398222016,"594":0.033378630768555094,"595":0.02713126702027569,"596":0.04961871018713406,"597":-0.14728937680492643,"598":0.021390684580403267,"599":-0.09796818084746901,"600":0.07623121883329831,"601":0.007242015777386073,"602":-0.030553037704431265,"603":-0.01703366618064854,"604":0.015160043458907176,"605":-0.002116366012058978,"606":-0.012410503819570665,"607":0.061622375265526906,"608":0.02698861602385835,"609":0.024638283444019114,"610":-0.00903464898843036,"611":0.047458366249197345,"612":-0.05041641476068838,"613":-0.019853405558559342,"614":0.09858114465092904,"615":0.004514554215268655,"616":0.01722276864264976,"617":0.009674131699250292,"618":-0.02078401412527328,"619":-0.2535434767374491,"620":0.04053409686530791,"621":-0.02126846912683975,"622":0.015216114904158726,"623":0.04390100320199955,"624":0.2682538789376074,"625":-3.7976836883230147,"626":-0.04063519701183187,"627":0.9810015158953231,"628":-0.039075905197036916,"629":0.032548379789154404,"630":-0.009800284838107314,"631":-0.03895556926152241,"632":0.05138791292783347,"633":0.04782626073072754,"634":0.05241453632806194,"635":-0.0276661296209143,"636":-0.029246883357555313,"637":-0.040784998334985154,"638":0.07824118643921334,"639":0.03682190738406791,"640":-0.030191365947786698,"641":-0.009610225227162218,"642":0.05603700453820855,"643":0.005838966536500328,"644":0.014719342392066764,"645":0.05386749692238042,"646":0.002455980206504655,"647":0.0805382073101051,"648":0.03798552282681292,"649":-0.05425205798910308,"650":0.055079861348507626,"651":-0.023699889664595367,"652":0.0007268771256014229,"653":-0.004137936572136668,"654":0.07245227827342118,"655":0.15850558605131346,"656":0.05839612764622274,"657":-0.012789544513982837,"658":0.020361013636282028,"659":0.034923870060016696,"660":0.06366949637704042,"661":0.044707563035547015,"662":0.5571849074658864,"663":0.0456746202790291,"664":0.03614810234026599,"665":-0.018635716926749646,"666":-0.04205203847080797,"667":-0.07415801385536062,"668":-0.050958681735348746,"669":0.03263917147995722,"670":0.01830785670009625,"671":-0.028618384542083857,"672":1.2172909473209668,"673":-0.022929770512254127,"674":0.0023977332103559345,"675":-0.06572385125404717,"676":-0.035705925935862304,"677":-0.011590932599428621,"678":-0.040501038800171585,"679":-0.027531887662836076,"680":0.022888410982198112,"681":0.012191252300093708,"682":0.09455995649615494,"683":0.047389925425147375,"684":0.009068794833758753,"685":0.05577974853017994,"686":-0.015193948689231802,"687":-0.20371715320772188,"688":0.04409946255636202,"689":-0.04594705177530214,"690":0.0009398621417890764,"691":-0.034181473371341364,"692":0.0064548622226977305,"693":0.02433914779594766,"694":0.007623995893751681,"695":0.026971593713793183,"696":-0.00010473740910171313,"697":-0.06297708983672533,"698":-0.07979983587615354,"699":-0.024409498987025175,"700":-0.05298670603881754,"701":0.05864719442079795,"702":-0.04257917833329624,"703":0.1199366934832482,"704":0.03524510002792429,"705":4.307801122045616,"706":-0.1146591554191736,"707":-0.054293702103139656,"708":-0.020420866449476986,"709":-0.02286842445183953,"710":-0.14748958779108665,"711":-0.043768829201046466,"712":-0.05782896789572034,"713":0.04229069628561001,"714":-0.004003114635407431,"715":0.044513664148375987,"716":0.029639272603144678,"717":-0.17391242103454266,"718":-0.04459686603556116,"719":0.030423878235095608,"720":0.03225067466407737,"721":0.22105584823404378,"722":0.0004374240026096389,"723":0.04536588961676895,"724":0.017852063382950405,"725":-0.028338050910182717,"726":0.0810381546388696,"727":-0.01870600412079195,"728":-0.030877617075276104,"729":-0.021456187858641226,"730":-0.028217351802244554,"731":0.01976981162751614,"732":-0.029724711200470537,"733":0.10712894909631797,"734":-1.4088760909744316,"735":-0.011165019878700659,"736":-0.013283875452106205,"737":-0.05914003452343645,"738":-0.07559805527781571,"739":-0.5983120170044846,"740":-0.02746004485361631,"741":-0.04662749854188233,"742":0.07295629151640891,"743":-0.0026454564390428715,"744":-0.005545797906242107,"745":0.05854675757522383,"746":0.011017989944208372,"747":-4.336208258571896,"748":-0.022969686025800417,"749":0.05492992173220534,"750":-0.00812033299162308,"751":0.0836236477539234,"752":0.023084881014300878,"753":0.036249884687925615,"754":-0.06107599688654645,"755":0.0274480757213316,"756":-0.0006002180461069674,"757":0.061180563253244453,"758":-0.06651152719936372,"759":-0.01552486426071826,"760":0.004911702071961271,"761":-0.009455794412878462,"762":-0.05700971976305715,"763":0.10560721530521058,"764":-0.06611248212771519,"765":-0.04023762778698667,"766":-0.009650423502268447,"767":-0.0276111689333958,"768":-0.0690859778906386,"769":-0.02393911884316716,"770":-0.03135782072812769,"771":0.06669231402218881,"772":-0.02565281971564197,"773":-0.012427133309316019,"774":-0.05861879297891086,"775":0.7525430054062929,"776":-0.11454331492030816,"777":0.003768585143527439,"778":0.005204769562461669,"779":-0.046296725539362656,"780":0.024899851067030285,"781":-0.02894225565263218,"782":-0.005910625758392989,"783":-0.05074945896421307,"784":0.09811734675201109,"785":-0.011968740400079524,"786":0.032539438934589125,"787":-0.06901358536938658,"788":0.03215223605299547,"789":-0.06119161808867231,"790":0.020420344810902217,"791":0.02469122108194915,"792":0.0480834713507166,"793":0.02223612258999731,"794":0.040516235784406086,"795":-0.05129317954531475,"796":0.00858494464626855,"797":0.020638731643985393,"798":-0.03987820690140109,"799":0.08447433972679508,"800":0.051224977213633595,"801":-0.05365657380233607,"802":0.03504280783449313,"803":-0.07994708263309727,"804":0.030959886713909587,"805":-0.01022280258763573,"806":0.03384615238480797,"807":-0.03130704372852672,"808":-0.013182995536222741,"809":-0.02963535368954719,"810":-0.027818412553652715,"811":-0.05548218433571275,"812":-0.0211290954254549,"813":-0.03308672222510797,"814":0.019081217384512884,"815":-0.013701324172286623,"816":0.035327297425231045,"817":-0.031916048223196054,"818":-0.033967004946083554,"819":-0.428495523151777,"820":0.03735948003347177,"821":0.04018206789463241,"822":-0.021032363718300216,"823":0.0006054205560079183,"824":0.49742556943597893,"825":-3.57932497033455,"826":0.01921948722542733,"827":1.0551762261611561,"828":-0.0010733185877167,"829":0.0006801890461755061,"830":0.0008739930399992604,"831":0.011271318202548713,"832":0.07727626434012025,"833":-0.004264822211057064,"834":0.001992154429246292,"835":-0.028963666179474153,"836":-0.03931499755935624,"837":-0.04552697503636165,"838":0.043394786376832375,"839":0.025497417637674873,"840":0.007832472929573208,"841":0.011323930185943272,"842":-0.002586493170407389,"843":-0.01571180596951887,"844":0.04725445526442603,"845":-0.02870035683762509,"846":-0.006355262051363227,"847":0.035884410339508024,"848":0.013299812491691267,"849":-0.10860485763811542,"850":0.0779878094641272,"851":-0.023326613646791443,"852":0.02115219842173021,"853":-0.021524561202249104,"854":0.04110173494822971,"855":-0.005907595514300921,"856":0.07133790903969656,"857":-0.02803876020315482,"858":-0.0648575061072545,"859":-0.0039889905915851355,"860":0.04391642483885232,"861":0.03155308049913364,"862":0.44279753713151654,"863":-0.004875231091866304,"864":0.00409397288130298,"865":-0.009666073474755274,"866":0.00275991925700037,"867":0.010885467891185083,"868":-0.0050760585333352965,"869":0.01972110236374186,"870":-0.06147447384681875,"871":0.011917871033888895,"872":1.4325510163228015,"873":0.01925754299419084,"874":-0.0011157287967682543,"875":-0.031683682228036074,"876":-0.008683359036438219,"877":-0.04382320460635425,"878":-0.05172696541913613,"879":0.0036410720789398347,"880":-0.014565632856290044,"881":-0.02971167377545191,"882":0.05271879635827562,"883":0.032661466807233606,"884":-0.010327762988730518,"885":0.033974512144736194,"886":-0.005262562283460216,"887":-0.10105028653264328,"888":-0.008781652945921052,"889":-0.045960032200112876,"890":-0.046272671881222124,"891":-0.006147086837893969,"892":0.033615582163688804,"893":0.06891253800363477,"894":-0.018022573146349256,"895":0.024880555554605364,"896":-0.024421066597309197,"897":-0.037595954336599235,"898":-0.03537392965257955,"899":0.02373846854221173,"900":-0.04183067879957723,"901":0.0014292196610551872,"902":-0.07131202931534879,"903":0.33199150089276136,"904":-0.0003795546304640839,"905":4.941549942591909,"906":-0.06291125125534873,"907":0.010266520339594814,"908":0.06928715286039526,"909":-0.03704419887953892,"910":-0.20592603368542478,"911":-0.01965255681650664,"912":-0.04841343189388569,"913":-0.024952154897932125,"914":0.022984970756379192,"915":0.011636776623289493,"916":-0.013753974567750156,"917":-0.005192976114782768,"918":-0.002666505216303137,"919":0.015953526604716328,"920":0.010335062314155704,"921":0.28560747345483617,"922":0.03068602801700934,"923":0.015426906043744434,"924":-0.014754782232304852,"925":-0.05280665849681535,"926":0.06379231649769554,"927":0.0216652784325703,"928":0.00008830427593580002,"929":-0.03546464153725794,"930":-0.04706410467735943,"931":-0.0025686971560105023,"932":0.01215053961437793,"933":0.05167826341394828,"934":-1.4265283643911135,"935":-0.030341978674855056,"936":-0.006024674339554811,"937":-0.03025203416535918,"938":-0.019542923335625444,"939":-0.6649138936790778,"940":-0.006582537352972955,"941":0.0445162702186887,"942":0.00468269103548632,"943":0.06145779211775803,"944":-0.042003168292286175,"945":0.04055595418000544,"946":-0.015352010947956352,"947":-4.972395559349547,"948":-0.0027698473326636225,"949":0.01756722621292936,"950":-0.013068365336725968,"951":0.027234171288319124,"952":0.022081316585997778,"953":-0.007807477886946191,"954":-0.07313209094108356,"955":0.0022789604019362443,"956":-0.0007467427667864866,"957":0.017351934013273274,"958":-0.007101710631771496,"959":0.06752073985521667,"960":0.03662037862188515,"961":-0.04449380417412389,"962":-0.03216428118281355,"963":0.053110781997289185,"964":-0.056659413250390554,"965":-0.01433800448973034,"966":-0.023911268744164063,"967":-0.014497763564408964,"968":-0.05213798452560741,"969":-0.028668524363330065,"970":0.0038320707640611493,"971":-0.007882261913637237,"972":-0.014516159939572835,"973":0.028047477172950034,"974":-0.020555504420829863,"975":0.7312776176204729,"976":-0.014183257109537379,"977":-0.03781275819365853,"978":0.004385808175626993,"979":-0.029621301269610396,"980":0.039630087117846935,"981":-0.05664748113389578,"982":-0.0033903443930411435,"983":-0.0009253309329587824,"984":0.0773977259783166,"985":0.004929562254161918,"986":0.027569299064606548,"987":-0.0398155633866241,"988":0.06944353164338189,"989":-0.0335666075336336,"990":-0.019159076993153244,"991":0.03336272551438938,"992":0.011689993748606542,"993":0.04708779993035174,"994":-0.0010840255737872685,"995":-0.06535294147738706,"996":0.004488907608934988,"997":0.03458911031068902,"998":-0.014566226233292404,"999":0.11512864846442802}},"b2":{"n":5,"d":1,"w":{"0":3.9015918401100396,"1":2.638475534058713,"2":1.7936219626235794,"3":2.3823323588204777,"4":2.1723051280068373}}}});
        }
    }

    function initBombMap(map){
        for (var i = 0; i < map.width - 1; i++) {
            for (var j = 0; j < map.height - 1; j++) {
                if(bombMap[j] === undefined) {
                    bombMap[j] = [];
                }
                bombMap[j][i] = {value: 0, birth: 0};
            }
        }
    }

    var actions = [
        'right',
        'down',
        'left',
        'up',
        'stop',
        'bomb'
    ];

    var lastWins = 0;
    var lastLoses = 0;

    function isOnFire(player) {
        var x = Math.round(player.x);
        var y = Math.round(player.y);
        if (bombMap[y] && bombMap[y][x]) {
            return bombMap[y][x].value;
        }
        return 0;
    }

    // Функция бота.
    // На входе принимает данные о карте и других ботах/элементах
    // На выкоде команда к действию тип <string>

    // my_info  - информация об этом боте
    // my_state - Object в котором можно хранить временные данные бота
    // map - информация о карте и некоторые константы
    // map_objects - информация о временных объектах на карте. Таких как игроки, бомбы, магичесие артефакты
    function dergachevBot(my_info, my_state, map, map_objects, cursors) {
        var inputs = [];
        var {x, y} = my_info;

        if(!inited){
            initBombMap(map);
        }

        var width = map.width - 3;
        var height = map.height - 3;

        makeBombMap(map, map_objects);

        for (var i = y - 2; i < y + 3; i++){
            for (var j = x - 2; j < x + 3; j++){
                inputs.push(isOnFire({x: j, y: i}));
                // inputs.push(map(y, i) === map.wall ? 1 : 0);
                // console.log(inputs[inputs.length - 1]);
            }
        }

        inputs.push((x - 1) / width);
        inputs.push((y - 1) / height);


        var otherPlayer = {x: 0, y: 0};

        for (var p in map_objects) {
            var object = map_objects[p];
            if (object.type === 'player' ) {
                if (object.id === my_info.id) {
                    continue; // myself
                }
                otherPlayer = object;
            }
        }
        var leftOffset = (otherPlayer.x - x) / width;
        if (leftOffset === 0) {
            inputs.push(0);
            inputs.push(0);
        } else if (leftOffset < 0) {
            inputs.push(-leftOffset);
            inputs.push(1);
        } else if (leftOffset > 0) {
            inputs.push(1);
            inputs.push(leftOffset);
        }

        var topOffset = (otherPlayer.y - y) / height;
        if (topOffset === 0) {
            inputs.push(0);
            inputs.push(0);
        } else if (topOffset < 0) {
            inputs.push(-topOffset);
            inputs.push(1);
        } else if (topOffset > 0) {
            inputs.push(1);
            inputs.push(topOffset);
        }

        var score = 0;
        if(my_info.wins > lastWins){
            lastWins = my_info.wins;
            score += 10;
            console.log('win');
            //localStorage.setItem('brain', JSON.stringify(agent.toJSON()));
        }

        if (!inited) {
            init(inputs.length);
            inited = true;
        } else {
            //agent.learn(score);
        }

        //score += isOnFire(otherPlayer);
        score += 0.5 - isOnFire(my_info);

        console.log(score);

        var action = agent.act(inputs);
        return actions[action];
    }
})();
