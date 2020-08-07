import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { MostPopularvisiteurComponent } from './most-popular.component';

describe('MostPopularComponent', () => {
  let component: MostPopularvisiteurComponent;
  let fixture: ComponentFixture<MostPopularvisiteurComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ MostPopularvisiteurComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(MostPopularvisiteurComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
