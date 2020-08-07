import { Component, OnInit } from '@angular/core';
import { ProjetService } from '../../service/ImplementationService/projet.service'
import { HttpClient } from '@angular/common/http'
import { Task } from '../../service/Interfaces/task'
import { Info } from '../../service/Interfaces/info'
import { User } from 'src/app/service/Interfaces/user';

@Component({
  selector: 'app-chartsadmin',
  templateUrl: './charts.component.html',
  styleUrls: ['./charts.component.css']
})
export class ChartsComponentAdmin implements OnInit {

  tasks: Task[];
  users:User[];
  infos: Info[];
  data = new Array();
  count_by_categorie = new Array();
  totalrating_by_categorie = new Array();
  totalliking_by_categorie = new Array();
  rated_by_categorie = new Array();
  count_by_subcategorie_accessories = new Array();
  totalrating_by_subcategorie_accessories = new Array();
  totalliking_by_subcategorie_accessories = new Array();
  rated_by_subcategorie_accessories = new Array();
  count_by_subcategorie_apparel = new Array();
  totalrating_by_subcategorie_apparel = new Array();
  totalliking_by_subcategorie_apparel = new Array();
  rated_by_subcategorie_apparel = new Array();
  count_by_subcategorie_books = new Array();
  totalrating_by_subcategorie_books = new Array();
  totalliking_by_subcategorie_books = new Array();
  rated_by_subcategorie_books = new Array();
  count_by_subcategorie_footwear = new Array();
  totalrating_by_subcategorie_footwear = new Array();
  totalliking_by_subcategorie_footwear = new Array();
  rated_by_subcategorie_footwear = new Array();
  count_by_subcategorie_personal_care = new Array();
  totalrating_by_subcategorie_personal_care = new Array();
  totalliking_by_subcategorie_personal_care = new Array();
  rated_by_subcategorie_personal_care = new Array();
  count_by_subcategorie_phones = new Array();
  totalrating_by_subcategorie_phones = new Array();
  totalliking_by_subcategorie_phones = new Array();
  rated_by_subcategorie_phones = new Array();
  count_by_subcategorie_sporting = new Array();
  totalrating_by_subcategorie_sporting = new Array();
  totalliking_by_subcategorie_sporting = new Array();
  rated_by_subcategorie_sporting = new Array();
  nameproductsleastpopular = new Array();
  countactedtimeleastpopular =  new Array();

  constructor(private projetService: ProjetService, private http: HttpClient) { }
  public barChartOptions = {
    scaleShowVerticalLines: false,
    responsive: true
  };

  /* Create a Bar Chart */
  public barChartLabels = ['Accessories', 'Apparel', 'Books', 'Footwear', 'Personal Care', 'Phones', 'Sporting Goods'];
  public barChartType = 'bar';
  public barChartLegend = true;
  public barChartData = [
    {data: this.count_by_categorie, label: 'Count',backgroundColor: "#6bd098"},
    {data: this.totalrating_by_categorie, label: 'Sumrating'},
    {data: this.totalliking_by_categorie, label: 'Sumlikes'},
    {data: this.rated_by_categorie, label: 'rated',backgroundColor:"black"}
  ];

  leastpopularproducts (): void {
    this.projetService.leastpopularproducts().subscribe(tasks => (this.tasks = tasks))
  }

  leastactiveusers (): void {
    this.projetService.leastactiveusers().subscribe(users => (this.users = users))
  }

  delete (task: Task): void {
    this.tasks = this.tasks.filter(h => h !== task)
    this.projetService
        .deleteTask(task.produitId)
        .subscribe(() => console.log('Task Deleted'))
  }

  deleteuser (user: User): void {
    this.users = this.users.filter(h => h !== user)
    this.projetService
        .deleteUser(user.userId)
        .subscribe(() => console.log('User Deleted'))
  }

  ngOnInit() {
    this.leastactiveusers ()
    this.leastpopularproducts()

    this.projetService
      .info()
      .subscribe((data: Info[]) => {
        this.infos = data;
        this.infos.forEach(n => {
          this.count_by_categorie.push(n.accessoriescount);
          this.count_by_categorie.push(n.apparelcount);
          this.count_by_categorie.push(n.bookscount);
          this.count_by_categorie.push(n.footwearcount);
          this.count_by_categorie.push(n.personal_Carecount);
          this.count_by_categorie.push(n.phonescount);
          this.count_by_categorie.push(n.sportcount);
          
          this.totalrating_by_categorie.push(n.accessoriestotalrating);
          this.totalrating_by_categorie.push(n.appareltotalrating);
          this.totalrating_by_categorie.push(n.bookstotalrating);
          this.totalrating_by_categorie.push(n.footweartotalrating);
          this.totalrating_by_categorie.push(n.personal_Caretotalrating);
          this.totalrating_by_categorie.push(n.phonestotalrating);
          this.totalrating_by_categorie.push(n.sporttotalrating);

          this.totalliking_by_categorie.push(n.accessoriestotalliking);
          this.totalliking_by_categorie.push(n.appareltotalliking);
          this.totalliking_by_categorie.push(n.bookstotalliking);
          this.totalliking_by_categorie.push(n.footweartotalliking);
          this.totalliking_by_categorie.push(n.personal_Caretotalliking);
          this.totalliking_by_categorie.push(n.phonestotalliking);
          this.totalliking_by_categorie.push(n.sporttotalliking);

          this.rated_by_categorie.push(n.accessoriesrated);
          this.rated_by_categorie.push(n.apparelrated);
          this.rated_by_categorie.push(n.booksrated);
          this.rated_by_categorie.push(n.footwearrated);
          this.rated_by_categorie.push(n.personal_Carerated);
          this.rated_by_categorie.push(n.phonesrated);
          this.rated_by_categorie.push(n.sportrated);

          this.count_by_subcategorie_accessories.push(n.bags_Accessoriescount);
          this.count_by_subcategorie_accessories.push(n.beltscount);
          this.count_by_subcategorie_accessories.push(n.cufflinkscount);
          this.count_by_subcategorie_accessories.push(n.eyewear_Accessoriescount);
          this.count_by_subcategorie_accessories.push(n.giftcount);
          this.count_by_subcategorie_accessories.push(n.glovescount);
          this.count_by_subcategorie_accessories.push(n.headwear_Accessoriescount);
          this.count_by_subcategorie_accessories.push(n.jewellerycount);
          this.count_by_subcategorie_accessories.push(n.walletscount);
          this.count_by_subcategorie_accessories.push(n.watches_Accessoriescount);
          
          this.totalrating_by_subcategorie_accessories.push(n.bags_Accessoriestotalrating);
          this.totalrating_by_subcategorie_accessories.push(n.beltstotalrating);
          this.totalrating_by_subcategorie_accessories.push(n.cufflinkstotalrating);
          this.totalrating_by_subcategorie_accessories.push(n.eyewear_Accessoriestotalrating);
          this.totalrating_by_subcategorie_accessories.push(n.gifttotalrating);
          this.totalrating_by_subcategorie_accessories.push(n.glovestotalrating);
          this.totalrating_by_subcategorie_accessories.push(n.headwear_Accessoriestotalrating);
          this.totalrating_by_subcategorie_accessories.push(n.jewellerytotalrating);
          this.totalrating_by_subcategorie_accessories.push(n.walletstotalrating);
          this.totalrating_by_subcategorie_accessories.push(n.watches_Accessoriestotalrating);

          this.totalliking_by_subcategorie_accessories.push(n.bags_Accessoriestotalliking);
          this.totalliking_by_subcategorie_accessories.push(n.beltstotalliking);
          this.totalliking_by_subcategorie_accessories.push(n.cufflinkstotalliking);
          this.totalliking_by_subcategorie_accessories.push(n.eyewear_Accessoriestotalliking);
          this.totalliking_by_subcategorie_accessories.push(n.giftstotalliking);
          this.totalliking_by_subcategorie_accessories.push(n.glovestotalliking);
          this.totalliking_by_subcategorie_accessories.push(n.headwear_Accessoriestotalliking);
          this.totalliking_by_subcategorie_accessories.push(n.jewellerytotalliking);
          this.totalliking_by_subcategorie_accessories.push(n.walletstotalliking);
          this.totalliking_by_subcategorie_accessories.push(n.watches_Accessoriestotalliking);

          this.rated_by_subcategorie_accessories.push(n.bags_Accessoriesrated);
          this.rated_by_subcategorie_accessories.push(n.beltsrated);
          this.rated_by_subcategorie_accessories.push(n.cufflinksrated);
          this.rated_by_subcategorie_accessories.push(n.eyewear_Accessoriesrated);
          this.rated_by_subcategorie_accessories.push(n.giftsrated);
          this.rated_by_subcategorie_accessories.push(n.glovesrated);
          this.rated_by_subcategorie_accessories.push(n.headwear_Accessoriesrated);
          this.rated_by_subcategorie_accessories.push(n.jewelleryrated);
          this.rated_by_subcategorie_accessories.push(n.walletsrated);
          this.rated_by_subcategorie_accessories.push(n.watches_Accessoriesrated);

          this.count_by_subcategorie_apparel.push(n.innerwearcount);
          this.count_by_subcategorie_apparel.push(n.scarvescount);
          this.count_by_subcategorie_apparel.push(n.bottomwearcount);
          this.count_by_subcategorie_apparel.push(n.topwearcount);
          this.count_by_subcategorie_apparel.push(n.apparel_setcount);
          this.count_by_subcategorie_apparel.push(n.dresscount);
          this.count_by_subcategorie_apparel.push(n.tiescount);
          this.count_by_subcategorie_apparel.push(n.stolescount);
          this.count_by_subcategorie_apparel.push(n.loungewear_nightwearcount);
          this.count_by_subcategorie_apparel.push(n.sareecount);
          
          this.totalrating_by_subcategorie_apparel.push(n.innerweartotalrating);
          this.totalrating_by_subcategorie_apparel.push(n.scarvestotalrating);
          this.totalrating_by_subcategorie_apparel.push(n.bottomweatotalrrating);
          this.totalrating_by_subcategorie_apparel.push(n.topweartotalrating);
          this.totalrating_by_subcategorie_apparel.push(n.apparel_settotalrating);
          this.totalrating_by_subcategorie_apparel.push(n.dresstotalrating);
          this.totalrating_by_subcategorie_apparel.push(n.tiestotalrating);
          this.totalrating_by_subcategorie_apparel.push(n.stolestotalrating);
          this.totalrating_by_subcategorie_apparel.push(n.loungewear_nightweartotalrating);
          this.totalrating_by_subcategorie_apparel.push(n.sareetotalrating);

          this.totalliking_by_subcategorie_apparel.push(n.innerweartotalliking);
          this.totalliking_by_subcategorie_apparel.push(n.scarvestotalliking);
          this.totalliking_by_subcategorie_apparel.push(n.bottomweartotalliking);
          this.totalliking_by_subcategorie_apparel.push(n.topweartotalliking);
          this.totalliking_by_subcategorie_apparel.push(n.apparel_settotalliking);
          this.totalliking_by_subcategorie_apparel.push(n.dresstotalliking);
          this.totalliking_by_subcategorie_apparel.push(n.tiestotalliking);
          this.totalliking_by_subcategorie_apparel.push(n.stolestotalliking);
          this.totalliking_by_subcategorie_apparel.push(n.loungewear_nightweartotalliking);
          this.totalliking_by_subcategorie_apparel.push(n.sareetotalliking);

          this.rated_by_subcategorie_apparel.push(n.innerwearrated);
          this.rated_by_subcategorie_apparel.push(n.scarvesrated);
          this.rated_by_subcategorie_apparel.push(n.bottomwearrated);
          this.rated_by_subcategorie_apparel.push(n.topwearrated);
          this.rated_by_subcategorie_apparel.push(n.apparel_setrated);
          this.rated_by_subcategorie_apparel.push(n.dressrated);
          this.rated_by_subcategorie_apparel.push(n.tiesrated);
          this.rated_by_subcategorie_apparel.push(n.stolesrated);
          this.rated_by_subcategorie_apparel.push(n.loungewear_nightwearrated);
          this.rated_by_subcategorie_apparel.push(n.sareerated);

          this.count_by_subcategorie_books.push(n.generalcount);
          this.count_by_subcategorie_books.push(n.childrencount);
          this.count_by_subcategorie_books.push(n.classicscount);
          this.count_by_subcategorie_books.push(n.crimecount);
          this.count_by_subcategorie_books.push(n.families_relationshipcount);
          this.count_by_subcategorie_books.push(n.fantasycount);
          this.count_by_subcategorie_books.push(n.literaturecount);
          this.count_by_subcategorie_books.push(n.otherscount);
          this.count_by_subcategorie_books.push(n.romancecount);
          this.count_by_subcategorie_books.push(n.science_fictioncount);
          this.count_by_subcategorie_books.push(n.short_storiescount);
          this.count_by_subcategorie_books.push(n.young_adultscount);

          this.totalrating_by_subcategorie_books.push(n.generaltotalrating);
          this.totalrating_by_subcategorie_books.push(n.childrentotalrating);
          this.totalrating_by_subcategorie_books.push(n.classicstotalrating);
          this.totalrating_by_subcategorie_books.push(n.crimetotalrating);
          this.totalrating_by_subcategorie_books.push(n.families_relationshiptotalrating);
          this.totalrating_by_subcategorie_books.push(n.fantasytotalrating);
          this.totalrating_by_subcategorie_books.push(n.literaturetotalrating);
          this.totalrating_by_subcategorie_books.push(n.otherstotalrating);
          this.totalrating_by_subcategorie_books.push(n.romancetotalrating);
          this.totalrating_by_subcategorie_books.push(n.science_fictiontotalrating);
          this.totalrating_by_subcategorie_books.push(n.short_storiestotalrating);
          this.totalrating_by_subcategorie_books.push(n.young_adultstotalrating);

          this.totalliking_by_subcategorie_books.push(n.generaltotalliking);
          this.totalliking_by_subcategorie_books.push(n.childrentotalliking);
          this.totalliking_by_subcategorie_books.push(n.classicstotalliking);
          this.totalliking_by_subcategorie_books.push(n.crimetotalliking);
          this.totalliking_by_subcategorie_books.push(n.families_relationshiptotalliking);
          this.totalliking_by_subcategorie_books.push(n.fantasytotalliking);
          this.totalliking_by_subcategorie_books.push(n.literaturetotalliking);
          this.totalliking_by_subcategorie_books.push(n.otherstotalliking);
          this.totalliking_by_subcategorie_books.push(n.romancetotalliking);
          this.totalliking_by_subcategorie_books.push(n.science_fictiontotalliking);
          this.totalliking_by_subcategorie_books.push(n.short_storiestotalliking);
          this.totalliking_by_subcategorie_books.push(n.young_adultstotalliking);

          this.rated_by_subcategorie_books.push(n.generalrated);
          this.rated_by_subcategorie_books.push(n.childrenrated);
          this.rated_by_subcategorie_books.push(n.classicsrated);
          this.rated_by_subcategorie_books.push(n.crimerated);
          this.rated_by_subcategorie_books.push(n.families_relationshiprated);
          this.rated_by_subcategorie_books.push(n.fantasyrated);
          this.rated_by_subcategorie_books.push(n.literaturerated);
          this.rated_by_subcategorie_books.push(n.othersrated);
          this.rated_by_subcategorie_books.push(n.romancerated);
          this.rated_by_subcategorie_books.push(n.science_fictionrated);
          this.rated_by_subcategorie_books.push(n.short_storiesrated);
          this.rated_by_subcategorie_books.push(n.young_adultsrated);

          this.count_by_subcategorie_footwear.push(n.flip_flopscount);
          this.count_by_subcategorie_footwear.push(n.sandalcount);
          this.count_by_subcategorie_footwear.push(n.shoe_accessoriescount);
          this.count_by_subcategorie_footwear.push(n.shoescount);
          this.count_by_subcategorie_footwear.push(n.sockscount);

          this.totalrating_by_subcategorie_footwear.push(n.flip_flopstotalrating);
          this.totalrating_by_subcategorie_footwear.push(n.sandaltotalrating);
          this.totalrating_by_subcategorie_footwear.push(n.shoe_accessoriestotalrating);
          this.totalrating_by_subcategorie_footwear.push(n.shoestotalrating);
          this.totalrating_by_subcategorie_footwear.push(n.sockstotalrating);

          this.totalliking_by_subcategorie_footwear.push(n.flip_flopstotalliking);
          this.totalliking_by_subcategorie_footwear.push(n.sandaltotalliking);
          this.totalliking_by_subcategorie_footwear.push(n.shoe_accessoriestotalliking);
          this.totalliking_by_subcategorie_footwear.push(n.shoestotalliking);
          this.totalliking_by_subcategorie_footwear.push(n.sockstotalliking);

          this.rated_by_subcategorie_footwear.push(n.flip_flopsrated);
          this.rated_by_subcategorie_footwear.push(n.sandalrated);
          this.rated_by_subcategorie_footwear.push(n.shoe_accessoriesrated);
          this.rated_by_subcategorie_footwear.push(n.shoesrated);
          this.rated_by_subcategorie_footwear.push(n.socksrated);

          this.count_by_subcategorie_personal_care.push(n.bath_bodycount);
          this.count_by_subcategorie_personal_care.push(n.eyescount);
          this.count_by_subcategorie_personal_care.push(n.fragrancecount);
          this.count_by_subcategorie_personal_care.push(n.haircount);
          this.count_by_subcategorie_personal_care.push(n.lipscount);
          this.count_by_subcategorie_personal_care.push(n.makeupcount);
          this.count_by_subcategorie_personal_care.push(n.nailscount);
          this.count_by_subcategorie_personal_care.push(n.skin_carecount);

          this.totalrating_by_subcategorie_personal_care.push(n.bath_bodytotalrating);
          this.totalrating_by_subcategorie_personal_care.push(n.eyestotalrating);
          this.totalrating_by_subcategorie_personal_care.push(n.fragrancetotalrating);
          this.totalrating_by_subcategorie_personal_care.push(n.hairtotalrating);
          this.totalrating_by_subcategorie_personal_care.push(n.lipstotalrating);
          this.totalrating_by_subcategorie_personal_care.push(n.makeuptotalrating);
          this.totalrating_by_subcategorie_personal_care.push(n.nailstotalrating);
          this.totalrating_by_subcategorie_personal_care.push(n.skin_caretotalrating);

          this.totalliking_by_subcategorie_personal_care.push(n.bath_bodytotalliking);
          this.totalliking_by_subcategorie_personal_care.push(n.eyestotalliking);
          this.totalliking_by_subcategorie_personal_care.push(n.fragrancetotalliking);
          this.totalliking_by_subcategorie_personal_care.push(n.hairtotalliking);
          this.totalliking_by_subcategorie_personal_care.push(n.lipstotalliking);
          this.totalliking_by_subcategorie_personal_care.push(n.makeuptotalliking);
          this.totalliking_by_subcategorie_personal_care.push(n.nailstotalliking);
          this.totalliking_by_subcategorie_personal_care.push(n.skin_caretotalliking);

          this.rated_by_subcategorie_personal_care.push(n.bath_bodyrated);
          this.rated_by_subcategorie_personal_care.push(n.eyesrated);
          this.rated_by_subcategorie_personal_care.push(n.fragrancerated);
          this.rated_by_subcategorie_personal_care.push(n.hairrated);
          this.rated_by_subcategorie_personal_care.push(n.lipsrated);
          this.rated_by_subcategorie_personal_care.push(n.makeuprated);
          this.rated_by_subcategorie_personal_care.push(n.nailsrated);
          this.rated_by_subcategorie_personal_care.push(n.skin_carerated);
          
          this.count_by_subcategorie_phones.push(n.androidcount);
          this.count_by_subcategorie_phones.push(n.applecount);
          this.count_by_subcategorie_phones.push(n.honorcount);
          this.count_by_subcategorie_phones.push(n.huaweicount);
          this.count_by_subcategorie_phones.push(n.nokiacount);
          this.count_by_subcategorie_phones.push(n.onepluscount);
          this.count_by_subcategorie_phones.push(n.oppocount);
          this.count_by_subcategorie_phones.push(n.redmicount);
          this.count_by_subcategorie_phones.push(n.samsungcount);
          this.count_by_subcategorie_phones.push(n.sonycount);
          this.count_by_subcategorie_phones.push(n.vivocount);

          this.totalrating_by_subcategorie_phones.push(n.androidtotalrating);
          this.totalrating_by_subcategorie_phones.push(n.appletotalrating);
          this.totalrating_by_subcategorie_phones.push(n.honortotalrating);
          this.totalrating_by_subcategorie_phones.push(n.huaweitotalrating);
          this.totalrating_by_subcategorie_phones.push(n.nokiatotalrating);
          this.totalrating_by_subcategorie_phones.push(n.oneplustotalrating);
          this.totalrating_by_subcategorie_phones.push(n.oppototalrating);
          this.totalrating_by_subcategorie_phones.push(n.redmitotalrating);
          this.totalrating_by_subcategorie_phones.push(n.samsungtotalrating);
          this.totalrating_by_subcategorie_phones.push(n.sonytotalrating);
          this.totalrating_by_subcategorie_phones.push(n.vivototalrating);

          this.totalliking_by_subcategorie_phones.push(n.androidtotalliking);
          this.totalliking_by_subcategorie_phones.push(n.appletotalliking);
          this.totalliking_by_subcategorie_phones.push(n.honortotalliking);
          this.totalliking_by_subcategorie_phones.push(n.huaweitotalliking);
          this.totalliking_by_subcategorie_phones.push(n.nokiatotalliking);
          this.totalliking_by_subcategorie_phones.push(n.oneplustotalliking);
          this.totalliking_by_subcategorie_phones.push(n.oppototalliking);
          this.totalliking_by_subcategorie_phones.push(n.redmitotalliking);
          this.totalliking_by_subcategorie_phones.push(n.samsungtotalliking);
          this.totalliking_by_subcategorie_phones.push(n.sonytotalliking);
          this.totalliking_by_subcategorie_phones.push(n.vivototalliking);

          this.rated_by_subcategorie_phones.push(n.androidrated);
          this.rated_by_subcategorie_phones.push(n.applerated);
          this.rated_by_subcategorie_phones.push(n.honorrated);
          this.rated_by_subcategorie_phones.push(n.huaweirated)
          this.rated_by_subcategorie_phones.push(n.nokiarated);
          this.rated_by_subcategorie_phones.push(n.oneplusrated);
          this.rated_by_subcategorie_phones.push(n.opporated);
          this.rated_by_subcategorie_phones.push(n.redmirated);
          this.rated_by_subcategorie_phones.push(n.samsungrated);
          this.rated_by_subcategorie_phones.push(n.sonyrated);
          this.rated_by_subcategorie_phones.push(n.vivorated);
    
          this.count_by_subcategorie_sporting.push(n.bags_Sportingcount);
          this.count_by_subcategorie_sporting.push(n.eyewear_Sportingcount);
          this.count_by_subcategorie_sporting.push(n.headwear_Sportingcount);
          this.count_by_subcategorie_sporting.push(n.sports_equipmentcount);
          this.count_by_subcategorie_sporting.push(n.watches_Sportingcount);
          this.count_by_subcategorie_sporting.push(n.water_bottlecount);

          this.totalrating_by_subcategorie_sporting.push(n.bags_Sportingtotalrating);
          this.totalrating_by_subcategorie_sporting.push(n.eyewear_Sportingtotalrating);
          this.totalrating_by_subcategorie_sporting.push(n.headwear_Sportingtotalrating);
          this.totalrating_by_subcategorie_sporting.push(n.sports_equipmenttotalrating);
          this.totalrating_by_subcategorie_sporting.push(n.watches_Sportingtotalrating);
          this.totalrating_by_subcategorie_sporting.push(n.water_bottletotalrating);

          this.totalliking_by_subcategorie_sporting.push(n.bags_Sportingtotalliking);
          this.totalliking_by_subcategorie_sporting.push(n.eyewear_Sportingtotalliking);
          this.totalliking_by_subcategorie_sporting.push(n.headwear_Sportingtotalliking);
          this.totalliking_by_subcategorie_sporting.push(n.sports_equipmenttotalliking);
          this.totalliking_by_subcategorie_sporting.push(n.watches_Sportingtotalliking);
          this.totalliking_by_subcategorie_sporting.push(n.water_bottletotalliking);

          this.rated_by_subcategorie_sporting.push(n.bags_Sportingrated);
          this.rated_by_subcategorie_sporting.push(n.eyewear_Sportingrated);
          this.rated_by_subcategorie_sporting.push(n.headwear_Sportingrated);
          this.rated_by_subcategorie_sporting.push(n.sports_equipmentrated);
          this.rated_by_subcategorie_sporting.push(n.watches_Sportingrated);
          this.rated_by_subcategorie_sporting.push(n.water_bottlerated);       
      });
      


      this.count_by_categorie = this.count_by_categorie.filter(f => f !== undefined)
      this.totalrating_by_categorie = this.totalrating_by_categorie.sort((a, b) => {if(a !== undefined){return 1;}else{return 0;}})
      this.totalliking_by_categorie = this.totalliking_by_categorie.sort((a, b) => {if(a !== undefined){return 1;}else{return 0;}})
      this.rated_by_categorie = this.rated_by_categorie.sort((a, b) => {if(a !== undefined){return 1;}else{return 0;}})
      this.count_by_subcategorie_accessories = this.count_by_subcategorie_accessories.filter(f => f !== undefined)
      this.totalrating_by_subcategorie_accessories = this.totalrating_by_subcategorie_accessories.filter(f => f !== undefined)
      this.totalliking_by_subcategorie_accessories = this.totalliking_by_subcategorie_accessories.filter(f => f !== undefined)
      this.rated_by_subcategorie_accessories = this.rated_by_subcategorie_accessories.filter(f => f !== undefined)
      this.count_by_subcategorie_apparel = this.count_by_subcategorie_apparel.filter(f => f !== undefined)
      this.totalrating_by_subcategorie_apparel = this.totalrating_by_subcategorie_apparel.filter(f => f !== undefined)
      this.totalliking_by_subcategorie_apparel = this.totalliking_by_subcategorie_apparel.filter(f => f !== undefined)
      this.rated_by_subcategorie_apparel = this.rated_by_subcategorie_apparel.filter(f => f !== undefined)
      this.count_by_subcategorie_books = this.count_by_subcategorie_books.filter(f => f !== undefined)
      this.totalrating_by_subcategorie_books = this.totalrating_by_subcategorie_books.filter(f => f !== undefined)
      this.totalliking_by_subcategorie_books = this.totalliking_by_subcategorie_books.filter(f => f !== undefined)
      this.rated_by_subcategorie_books = this.rated_by_subcategorie_books.filter(f => f !== undefined)
      this.count_by_subcategorie_footwear = this.count_by_subcategorie_footwear.filter(f => f !== undefined)
      this.totalrating_by_subcategorie_footwear = this.totalrating_by_subcategorie_footwear.filter(f => f !== undefined)
      this.totalliking_by_subcategorie_footwear = this.totalliking_by_subcategorie_footwear.filter(f => f !== undefined)
      this.rated_by_subcategorie_footwear = this.rated_by_subcategorie_footwear.filter(f => f !== undefined)
      this.count_by_subcategorie_personal_care = this.count_by_subcategorie_personal_care.filter(f => f !== undefined)
      this.totalrating_by_subcategorie_personal_care = this.totalrating_by_subcategorie_personal_care.filter(f => f !== undefined)
      this.totalliking_by_subcategorie_personal_care = this.totalliking_by_subcategorie_personal_care.filter(f => f !== undefined)
      this.rated_by_subcategorie_personal_care = this.rated_by_subcategorie_personal_care.filter(f => f !== undefined)
      this.count_by_subcategorie_phones = this.count_by_subcategorie_phones.filter(f => f !== undefined)
      this.totalrating_by_subcategorie_phones = this.totalrating_by_subcategorie_phones.filter(f => f !== undefined)
      this.totalliking_by_subcategorie_phones = this.totalliking_by_subcategorie_phones.filter(f => f !== undefined)
      this.rated_by_subcategorie_phones = this.rated_by_subcategorie_phones.filter(f => f !== undefined)
      this.count_by_subcategorie_sporting = this.count_by_subcategorie_sporting.filter(f => f !== undefined)
      this.totalrating_by_subcategorie_sporting = this.totalrating_by_subcategorie_sporting.filter(f => f !== undefined)
      this.totalliking_by_subcategorie_sporting = this.totalliking_by_subcategorie_sporting.filter(f => f !== undefined)
      this.rated_by_subcategorie_sporting = this.rated_by_subcategorie_sporting.filter(f => f !== undefined)
      
      console.log(this.count_by_subcategorie_accessories)
      console.log(this.totalrating_by_categorie)
      console.log(this.count_by_subcategorie_books)
      console.log(this.count_by_subcategorie_footwear)
      console.log(this.count_by_subcategorie_personal_care)
      console.log(this.count_by_subcategorie_phones)
      console.log(this.count_by_subcategorie_sporting)
     });
  }
  
  
 
  /* Create a doughnut Chart to display the Most Liked Category*/
  public doughnutChartLabels = ['Accessories', 'Apparel', 'Books', 'Footwear', 'Personal Care', 'Phones', 'Sporting Goods'];
  public doughnutChartData = this.totalliking_by_categorie;
  public doughnutChartColors: any[] = [{ backgroundColor : ["#00CC66", "#00FF80","#33FF99","#99FFCC","#FFB266", "#B2FF66","#66FFB2"]}];
  public doughnutChartType = 'doughnut';
  

  /* Create a pie Chart to display the Most Posted Category*/
  public pieChartLabels = ['Accessories', 'Apparel', 'Books', 'Footwear', 'Personal Care', 'Phones', 'Sporting Goods'];
  public pieChartData = this.count_by_categorie;
  public pieChartColors: any[] = [{ backgroundColor : ["#FFCCCC", "#FFFFCC","#CCFFFF","#FFCCFF","#E0E0E0", "#99CCFF","#FF9999"]}];
  public pieChartType = 'pie';
}
