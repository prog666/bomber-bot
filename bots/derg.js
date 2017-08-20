(function isolate(){
    var Neuroevolution = function(options){
        var self = this;
        self.options = {
            activation:function(a){
                ap = (-a)/1;
                return (1/(1 + Math.exp(ap)))
            },
            randomClamped:function(){
                return Math.random() * 2 - 1;
            },
            population:50,
            elitism:0.2,
            randomBehaviour:0.2,
            mutationRate:0.1,
            mutationRange:0.5,
            network:[1, [1], 1],
            historic:0,
            lowHistoric:false,
            scoreSort:-1,
            nbChild:1
        }

        self.set = function(options){
            for(var i in options){
                if(this.options[i] != undefined){
                    self.options[i] = options[i];
                }
            }
        }

        self.set(options);

        //NEURON
        var Neuron = function(){
            this.value = 0;
            this.weights = [];
        }
        Neuron.prototype.populate = function(nb){
            this.weights = [];
            for(var i = 0; i < nb; i++){
                this.weights.push(self.options.randomClamped());
            }
        }
        //LAYER
        var Layer = function(index){
            this.id = index || 0;
            this.neurons = [];
        }
        Layer.prototype.populate = function(nbNeurons, nbInputs){
            this.neurons = [];
            for(var i = 0; i < nbNeurons; i++){
                var n = new Neuron();
                n.populate(nbInputs);
                this.neurons.push(n);
            }
        }
        //NETWORK
        var Network = function(){
            this.layers = [];
        }

        Network.prototype.perceptronGeneration = function(input, hiddens, output){
            var index = 0;
            var previousNeurons = 0;
            var layer = new Layer(index);
            layer.populate(input, previousNeurons);
            previousNeurons = input;
            this.layers.push(layer);
            index++;
            for(var i in hiddens){
                var layer = new Layer(index);
                layer.populate(hiddens[i], previousNeurons);
                previousNeurons = hiddens[i];
                this.layers.push(layer);
                index++;
            }
            var layer = new Layer(index);
            layer.populate(output, previousNeurons);
            this.layers.push(layer);
        }


        Network.prototype.getSave = function(){
            var datas = {
                neurons:[],
                weights:[]
            };
            for(var i in this.layers){
                datas.neurons.push(this.layers[i].neurons.length);
                for(var j in this.layers[i].neurons){
                    for(var k in this.layers[i].neurons[j].weights){
                        datas.weights.push(this.layers[i].neurons[j].weights[k]);
                    }
                }
            }
            return datas;
        }


        Network.prototype.setSave = function(save){
            var previousNeurons = 0;
            var index = 0;
            var indexWeights = 0;
            this.layers = [];
            for(var i in save.neurons){
                var layer = new Layer(index);
                layer.populate(save.neurons[i], previousNeurons);
                for(var j in layer.neurons){
                    for(var k in layer.neurons[j].weights){
                        layer.neurons[j].weights[k] = save.weights[indexWeights];
                        indexWeights++;
                    }
                }
                previousNeurons = save.neurons[i];
                index++;
                this.layers.push(layer);
            }
        }

        Network.prototype.compute = function(inputs){
            for(var i in inputs){
                if(this.layers[0] && this.layers[0].neurons[i]){
                    this.layers[0].neurons[i].value = inputs[i];
                }
            }

            var prevLayer = this.layers[0];
            for(var i = 1; i < this.layers.length; i++){
                for(var j in this.layers[i].neurons){
                    var sum = 0;
                    for(var k in prevLayer.neurons){
                        sum += prevLayer.neurons[k].value * this.layers[i].neurons[j].weights[k];
                    }
                    this.layers[i].neurons[j].value = self.options.activation(sum);
                }
                prevLayer = this.layers[i];
            }

            var out = [];
            var lastLayer = this.layers[this.layers.length - 1];
            for(var i in lastLayer.neurons){
                out.push(lastLayer.neurons[i].value);
            }
            return out;
        }
        //GENOM
        var Genome = function(score, network){
            this.score = score || 0;
            this.network = network || null;
        }
        //GENERATION
        var Generation = function(){
            this.genomes = [];
        }

        Generation.prototype.addGenome = function(genome){
            for(var i = 0; i < this.genomes.length; i++){
                if(self.options.scoreSort < 0){
                    if(genome.score > this.genomes[i].score){
                        break;
                    }
                }else{
                    if(genome.score < this.genomes[i].score){
                        break;
                    }
                }

            }
            this.genomes.splice(i, 0, genome);
        }

        Generation.prototype.breed = function(g1, g2, nbChilds){
            var datas = [];
            for(var nb = 0; nb < nbChilds; nb++){
                var data = JSON.parse(JSON.stringify(g1));
                for(var i in g2.network.weights){
                    if(Math.random() <= 0.5){
                        data.network.weights[i] = g2.network.weights[i];
                    }
                }

                for(var i in data.network.weights){
                    if(Math.random() <= self.options.mutationRate){
                        data.network.weights[i] += Math.random() * self.options.mutationRange * 2 - self.options.mutationRange;
                    }
                }
                datas.push(data);
            }

            return datas;
        }

        Generation.prototype.generateNextGeneration = function(){
            var nexts = [];

            for(var i = 0; i < Math.round(self.options.elitism * self.options.population); i++){
                if(nexts.length < self.options.population){
                    nexts.push(JSON.parse(JSON.stringify(this.genomes[i].network)));
                }
            }

            for(var i = 0; i < Math.round(self.options.randomBehaviour * self.options.population); i++){
                var n = JSON.parse(JSON.stringify(this.genomes[0].network));
                for(var k in n.weights){
                    n.weights[k] = self.options.randomClamped();
                }
                if(nexts.length < self.options.population){
                    nexts.push(n);
                }
            }

            var max = 0;
            while(true){
                for(var i = 0; i < max; i++){
                    var childs = this.breed(this.genomes[i], this.genomes[max], (self.options.nbChild > 0 ? self.options.nbChild : 1) );
                    for(var c in childs){
                        nexts.push(childs[c].network);
                        if(nexts.length >= self.options.population){
                            return nexts;
                        }
                    }
                }
                max++;
                if(max >= this.genomes.length - 1){
                    max = 0;
                }
            }
        }
        //GENERATIONS
        var Generations = function(){
            this.generations = [];
            var currentGeneration = new Generation();
        }

        Generations.prototype.firstGeneration = function(input, hiddens, output){
            var out = [];
            for(var i = 0; i < self.options.population; i++){
                var nn = new Network();
                nn.perceptronGeneration(self.options.network[0], self.options.network[1], self.options.network[2]);
                out.push(nn.getSave());
            }
            this.generations.push(new Generation());
            return out;
        }

        Generations.prototype.nextGeneration = function(){
            if(this.generations.length == 0){
                return false;
            }

            var gen = this.generations[this.generations.length - 1].generateNextGeneration();
            this.generations.push(new Generation());
            return gen;
        }


        Generations.prototype.addGenome = function(genome){
            if(this.generations.length == 0){
                return false;
            }

            return this.generations[this.generations.length - 1].addGenome(genome);
        }


        //SELF METHODS
        self.generations = new Generations();

        self.restart = function(){
            self.generations = new Generations();
        }

        self.nextGeneration = function(){
            var networks = [];
            if(self.generations.generations.length == 0){
                networks = self.generations.firstGeneration();
            }else{
                networks = self.generations.nextGeneration();
            }
            var nns = [];
            for(var i in networks){
                var nn = new Network();
                nn.setSave(networks[i]);
                nns.push(nn);
            }

            if(self.options.lowHistoric){
                if(self.generations.generations.length >= 2){
                    var genomes = self.generations.generations[self.generations.generations.length  - 2].genomes;
                    for(var i in genomes){
                        delete genomes[i].network;
                    }
                }
            }

            if(self.options.historic != -1){
                if(self.generations.generations.length > self.options.historic + 1){
                    self.generations.generations.splice(0, self.generations.generations.length - (self.options.historic + 1));
                }
            }
            return nns;
        }

        self.networkScore = function(network, score){
            self.generations.addGenome(new Genome(score, network.getSave()));
        }
    }

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
                    bomb_map.push({x: new_x, y: bomb.y, birth: bomb.birth, value: 1});
                }
                for(let bx = 1; bx <= radius; bx++){
                    let new_x = bomb.x - bx;
                    if(map(new_x, bomb.y) === WALL) {
                        break;
                    }
                    bomb_map.push({x: new_x, y: bomb.y, birth: bomb.birth, value: 1});
                }
                for(let by = 1; by <= radius; by++){
                    let new_y = bomb.y + by;
                    if(map(bomb.x, new_y) === WALL) {
                        break;
                    }
                    bomb_map.push({x: bomb.x, y: new_y, birth: bomb.birth, value: 1});
                }
                for(let by = 1; by <= radius; by++){
                    let new_y = bomb.y - by;
                    if(map(bomb.x, new_y) === WALL) {
                        break;
                    }
                    bomb_map.push({x: bomb.x, y: new_y, birth: bomb.birth, value: 1});
                }
            }
        }
        bomb_map.forEach(function ({x, y, birth, value}) {
            if (bombMap[y][x].birth !== birth) {
                bombMap[y][x].birth = birth;
                bombMap[y][x].value = value;
                var expireTime = BOMB_EXPLOSION_FINISH - Date.now() + birth;
                setTimeout(function () {
                    if (bombMap[y][x].birth === birth) {
                        bombMap[y][x].value = 0;
                        bombMap[y][x].birth = 0;
                    }
                }, expireTime);
            }
        });
    }

    addBot({
        name: "derg",
        routine: dergachevBot
    });

    var bombMap = [];

    var inited = false;
    // var agent;
    var index = 0;
    var score;
    var actions = [
        'right',
        'down',
        'left',
        'up',
        'stop',
        'bomb'
    ];

    var gens = [];

    var lastWins = 0;
    var lastLoses = 0;

    /*
    function init(size) {
        if (inited) {
            return;
        }
        inited = true;
        score = 0;
        console.log('initing');

        // create an environment object
        var env = {};
        env.getNumStates = function() { return size; };
        env.getMaxNumActions = function() { return 7; };

        // create the DQN agent
        var spec = {
            update: 'qlearn', // qlearn | sarsa
            gamma: 0.9, // discount factor, [0, 1)
            epsilon: 0.1, // initial epsilon for epsilon-greedy policy, [0, 1)
            alpha: 0.01, // value function learning rate
            experience_add_every: 5, // number of time steps before we add another experience to replay memory
            experience_size: 5000,  // size of experience replay memory
            learning_steps_per_iteration: 20,
            tderror_clamp: 1.0,
            num_hidden_units: 30
        };
        agent = new RL.DQNAgent(env, spec);
        var brain = localStorage.getItem('brain');
        if (brain) {
            agent.fromJSON(JSON.parse(brain));
        }
    }
    */

    var network;
    function init(size){
        let config = {
            population: 20,
            network: [size, [30, 15], 6],
            historic: 0,
            nbChild: 1
        }
        network = new Neuroevolution(config);
        gens = network.nextGeneration();
    }

    function initBombMap(map){
        for (var i = 0; i < map.width - 1; i++) {
            for (var j = 0; j < map.height - 1; j++) {
                if(bombMap[j] === undefined) {
                    bombMap[j] = [];
                }
                bombMap[j][i] = {birth: 0, value: 0};
            }
        }
    }

    function bombMapFn(x, y) {
        let realX = x;
        let realY = y;
        if (bombMap[realY] && bombMap[realY][realX]) {
            return bombMap[realY][realX].value ? 1 : 0;
        }
        return 0;
    }

    function getWallMap(map) {
        return (x, y) => {
            return map(x, y) === map.wall ? 1 : 0;
        }
    }

    function getMapValue(map, {x, y}) {
        let roundedX = Math.round(x);
        let roundedY = Math.round(y);
        let diffX = x - roundedX;
        let diffY = y - roundedY;

        let result = map(roundedX, roundedY);
        if (!result) {
            return result;
        }
        if (diffX > 0) {
            if (!map(roundedX + 1, roundedY)) {
                result -= Math.abs(diffX);
            }
        } else if (diffX < 0) {
            if (!map(roundedX - 1, roundedY)) {
                result -= Math.abs(diffX);
            }
        }

        if (diffY > 0) {
            if (!map(roundedX, roundedY + 1)) {
                result -= Math.abs(diffY);
            }
        } else if (diffY < 0) {
            if (!map(roundedX, roundedY - 1)) {
                result -= Math.abs(diffY)
            }
        }
        return result;
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

        for (var i = -3; i < 4; i++){
            for (var j = -3; j < 4; j++){
                let coords = {
                    x: x + j,
                    y: y + i,
                };
                let wall = getMapValue(getWallMap(map), coords);
                let fire = getMapValue(bombMapFn, coords);

                inputs.push(wall);
                inputs.push(fire);
            }
        }

        var otherPlayer = {x: 0, y: 0};
        var distanceToPlayer = 0;
        for (var p in map_objects) {
            var object = map_objects[p];
            if (object.type === 'player' ) {
                if (object.id === my_info.id) {
                    continue; // myself
                }
                otherPlayer = object;
            }
        }
        distanceToPlayer = 1 - ((Math.abs((otherPlayer.x - x) / width)) + (Math.abs((otherPlayer.y - y) / height)));
        var leftOffset = (otherPlayer.x - x) / width;
        inputs.push(leftOffset);

        var topOffset = (otherPlayer.y - y) / height;
        inputs.push(topOffset);

        if(my_info.wins > lastWins){
            lastWins = my_info.wins;
            score += 20;
            // agent.learn(score);
            localStorage.setItem('brain', JSON.stringify(agent.toJSON()));
        }
        if(my_info.loses > lastLoses){
            lastLoses = my_info.loses;
            // agent.learn(-100)
            score = 0;
        }

        init(inputs.length);

        let gen = gens[index];
        let res = gen.compute(inputs)
        maxIndex = 0;
        for (var i = 0; i < res.length; i++){
            if (res[i] > res[maxIndex]) {
                maxIndex = i;
            }
        }
        return actions[maxIndex];
        // var action = agent.act(inputs);
    }
})();
