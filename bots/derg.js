(function isolation(){
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

    addBot({
        name: "derg",
        routine: dergachevBot
    });

    /*
    positionX,
    positionY,
    map.width * map.height

    */

    var gens = [];
    var inited = false;
    var startTime = Date.now();
    var index = 0;

    var inputSize = 19;
    var network = new Neuroevolution({
        population: 20,
        network: [inputSize, [11], 3],
        historic: 5,
        nbChild: 5
    });
    function init(){
        gens = network.nextGeneration();
    }

    var actions = [
        'left',
        'right',
        'up',
        'down'
    ];

    var lastWins = 0;
    var lastLoses = 0;


    // Функция бота.
    // На входе принимает данные о карте и других ботах/элементах
    // На выкоде команда к действию тип <string>

    // my_info  - информация об этом боте
    // my_state - Object в котором можно хранить временные данные бота
    // map - информация о карте и некоторые константы
    // map_objects - информация о временных объектах на карте. Таких как игроки, бомбы, магичесие артефакты
    function dergachevBot(my_info, my_state, map, map_objects, cursors) {
        if (!inited) {
            inited = true;
            init();
        }
        var gen = gens[index];
        var {x, y} = my_info;

        var inputs = [
            //x / map.width,
            //y / map.height,
            map(x, y - 1) === map.wall ? 1 : 0,
            map(x, y + 1) === map.wall ? 1 : 0,
            map(x - 1, y) === map.wall ? 1 : 0,
            map(x + 1, y) === map.wall ? 1 : 0,
            0, // bomb exists
            0, // bomb left offset
            0, // bomb right offset
            0, // bomb top offset
            0, // bomb bottom offset
            0, // bomb exists 2nd
            0, // bomb left offset
            0, // bomb right offset
            0, // bomb top offset
            0, // bomb bottom offset
            0, // other player left offset
            0, // other player right offset
            0, // other player top offset
            0, // other player bottom offset
            my_info.bombRadius / 10,
        ];

        var bombs = 0;
        for (var p in map_objects) {
            var object = map_objects[p];
            if (object.type === 'player' ) {
                if (object.id === my_info.id) {
                    continue; // myself
                }
                var leftOffset = (object.x - x) / map.width;
                var topOffset = (object.y - y) / map.height;
                if (leftOffset === 0) {
                    inputs[14] = 0;
                    inputs[15] = 0;
                } else if (leftOffset < 0) {
                    inputs[14] = -leftOffset;
                    inputs[15] = 1;
                } else {
                    inputs[14] = 1;
                    inputs[15] = leftOffset;
                }
                if (topOffset === 0) {
                    inputs[16] = 0;
                    inputs[17] = 0;
                } else if (topOffset < 0) {
                    inputs[16] = -topOffset;
                    inputs[17] = 1;
                } else {
                    inputs[16] = 1;
                    inputs[17] =  topOffset;
                }

                // you can use info about other players:
                // object.id
                // object.type
                // object.x
                // object.y
                // object.lastAction
                // object.nextBombTime
                // object.speed
                // object.bombInterval
            }
            if (object.type === 'bomb') {
                inc = bombs * 5;
                inputs[4 + inc] = 1;
                var leftOffset = (object.x - x) / map.width;
                var topOffset = (object.y - y) / map.height;
                if (leftOffset === 0) {
                    inputs[5 + inc] = 0;
                    inputs[6 + inc] = 0;
                } else if (leftOffset < 0) {
                    inputs[5 + inc] = -leftOffset;
                    inputs[6 + inc] = 1;
                } else {
                    inputs[5 + inc] = 1;
                    inputs[6 + inc] = leftOffset;
                }
                if (topOffset === 0) {
                    inputs[7 + inc] = 0;
                    inputs[8 + inc] = 0;
                } else if (topOffset < 0) {
                    inputs[7 + inc] = -topOffset;
                    inputs[8 + inc] = 1;
                } else {
                    inputs[7 + inc] = 1;
                    inputs[8 + inc] =  topOffset;
                }
                bombs++;
            }
        }
        /*
        for (var i = 0; i < map.width; i++) {
            for (var j = 0; j < map.height; j++) {
                inputs.push(map(i, j) === map.wall ? 1 : 0);
            }
        }
        */

        var score = 0;
        var wins = my_info.wins > lastWins;
        var endGame = (my_info.loses > lastLoses || wins);
        if (my_info.loses > lastLoses) {
            lastLoses = my_info.loses;
        }
        if (wins) {
            lastWins = my_info.wins;
            score = 100000000000 / (Date.now() - startTime);
        } else {
            score = 1 - (1 / (Date.now() - startTime));
        }
        if (endGame) {
            startTime = Date.now();

            network.networkScore(gens[index], score);

            if (index === gens.length - 1) {
                index = 0;
                init();
            } else {
                index++;
            }
        }


        res = gen.compute(inputs);
        if (res[2] > .5 && Date.now() > my_info.nextBombTime) {
            return 'bomb';
        }
        if (res[0] > .75) {
            return 'left';
        }
        if (res[0] < .25) {
            return 'right';
        }
        if (res[1] > .75) {
            return 'up';
        }
        if (res[1] < .25) {
            return 'down';
        }
        return 'stop';
        return actions[Math.floor(res[0] * actions.length)];
        /*
        for (var i = 0; i < actions.length; i++){
            if (res[i] > .5) {
                return actions[i];
            }
        }
        return 'stop';
        */


        /*

        var x = Math.floor(my_info.x);
        var y = Math.floor(my_info.y);

        // my_info.type - типа == player
        // my_info.id - id игрока
        // my_info.x - координата на карте  в клетках
        // my_info.y - координата на карте  в клетках
        // my_info.lastAction - последнее известное действие
        // my_info.nextBombTime - timestamp когда сможет поставить следующую бомбу
        // my_info.speed - скорость игрока. в чем измеряется пока не понял))
        // my_info.bombInterval - как часто можно ставить бомбу

        // map.width - размерность карты в клетках
        // map.height - размерность карты в клетках
        // map.bombExpode - timestamp когда бомба взорвется;
        // map.bombVanish - timestamp когда бомба исчезнет после взрыва. можно проходить;

        // print info obout objects ->
        // var dbginfo = [];
        // map_objects.forEach(o => dbginfo.push(
        //     o.name + ':' + o.type +
        //     '(' + Math.round(o.x) + ',' + Math.round(o.y) +')'));
        // console.log(dbginfo.join(' - '));

        for (var p in map_objects) {
            var object = map_objects[p];
            if (object.type === 'player' ) {
                if (object.id === my_info.id) {
                    continue; // myself
                }

                // you can use info about other players:
                // object.id
                // object.type
                // object.x
                // object.y
                // object.lastAction
                // object.nextBombTime
                // object.speed
                // object.bombInterval
            }
            else if (object.type === 'bomb') {
                // object.birth
                // object.exists
                // object.expode
                // object.owner
                // object.type
                // object.vanish
                // object.x
                // object.y
            }
        }

        //  bombs
        if (my_state.bomb === undefined || my_state.bomb < Date.now() ) {
            my_state.bomb = Date.now() + my_info.bombInterval;
            return 'bomb';
        }

        // movements
        if(my_state.x === undefined || (my_state.x == x && my_state.y == y)){
            var r = route(map); // choose point to go
            my_state.x = r[0];
            my_state.y = r[1];
        }

        var distance_x = my_state.x - x;
        var distance_y = my_state.y - y;

        if (distance_y < 0) {
            if (map(x, y - 1) === map.wall) { // check if element above a wall
                if (map(x - 1, y) === map.wall) {
                    return 'right';
                } else {
                    return 'left';
                }
            }
            return 'up';
        }
        else if (distance_y > 0) {
            if (map(x, y + 1) === map.wall) { // check if element below a wall
                if (map(x - 1, y) === map.wall) {
                    return 'right';
                } else {
                    return 'left';
                }
            }
            return 'down';
        }
        else if (distance_x > 0) {
            if (map(x + 1, y) === map.wall) { // check if element below a wall
                if (map(x, y - 1) === map.wall) {
                    return 'down';
                } else {
                    return 'up';
                }
            }
            return 'right';
        }
        else if (distance_x < 0) {
            if (map(x - 1, y) === map.wall) { // check if element below a wall
                if (map(x, y - 1) === map.wall) {
                    return 'down';
                } else {
                    return 'up';
                }
            }
            return 'left';
        }

        // Бот может вернуть 6 действий строками
        // идти налево, направо, вверх, вниз, стоять, поставить бомбу.
        // left | right | up | down | stop | bomb
        return 'stop';
        */
    }
})();
